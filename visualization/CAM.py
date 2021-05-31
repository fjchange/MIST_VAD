import torch
import cv2
import numpy as np
from torch import nn

def softmaxed_weight_mlp(regressor):
    # calculate the softmax
    regressor_weights=[]
    for name,p in regressor.named_parameters():
        if 'weight' in name:
            regressor_weights.append(p)
    # for name,module in regressor.named_modules():
    #     print(name)
    # import pdb
    # pdb.set_trace()
    softmaxed_w=torch.softmax(regressor_weights[0],dim=-1)
    for w in regressor_weights[1:]:
        softmaxed_w=torch.mm(torch.softmax(w,dim=-1),softmaxed_w)
    return softmaxed_w

def get_CAM(feature_maps,softmaxed_w):
    # supoosed the feature map is original size
    # feature_maps is with shape [B,C,T,H,W]->[B,C,H,W]
    # softmaxed_w is with shape [B,C,2]
    # output is CAM map with shape [B,1,H,W]
    if feature_maps.shape.__len__()==5:
        feature_maps=torch.mean(feature_maps,dim=-3)
    elif feature_maps.shape.__len__()==4:
        feature_maps=feature_maps
    elif feature_maps.shape.__len__()==3:
        feature_maps=feature_maps.unsqueeze(0)
    else:
        raise ValueError('input feature maps should be [B,C,T,H,W] or [B,C,H,W] or [C,H,W], but get {}'.format(feature_maps.shape))
    # weighted_w[1,C,1,1]
    norm_CAM_maps=torch.sum(softmaxed_w[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*feature_maps,dim=1,keepdim=True)
    abnorm_CAM_maps=torch.sum(softmaxed_w[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*feature_maps,dim=1,keepdim=True)
    # min_max_norm the CAM_map to [0,1]
    norm_CAM_maps=norm_CAM_maps-norm_CAM_maps.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0]
    norm_CAM_maps=norm_CAM_maps/norm_CAM_maps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    abnorm_CAM_maps=abnorm_CAM_maps-abnorm_CAM_maps.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0]
    abnorm_CAM_maps=abnorm_CAM_maps/abnorm_CAM_maps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    return norm_CAM_maps,abnorm_CAM_maps

def visualize_CAM(CAM_map,origin_size):
    if isinstance(CAM_map,torch.Tensor):
        CAM_map = CAM_map.permute([1, 2, 0]).detach().cpu().numpy()
    elif isinstance(CAM_map,np.ndarray):
        CAM_map=CAM_map
    else:
        raise TypeError("CAM_map must be torch.Tensor or np.ndarray, but get {}".format(type(CAM_map)))
    CAM_map=cv2.resize((CAM_map*255.).astype(np.uint8),origin_size)

    return CAM_map

def test_CAM():
    Regressor=nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.6),
                            nn.Linear(512,32),nn.Dropout(0.6),nn.Linear(32,2))
    softmaxed_w=softmaxed_weight_mlp(Regressor)
    feature_maps=torch.zeros([5,1024,2,8,10])
    norm_CAM_maps,abnorm_CAM_maps=get_CAM(feature_maps,softmaxed_w)

    CAM_map=visualize_CAM(norm_CAM_maps[0],(320,240))
    cv2.imwrite('./cam.jpg',CAM_map)


def visualize_CAM_with_clip(CAM_map,clip,origin_size):
    CAM=visualize_CAM(CAM_map,origin_size)
    # clip is with shape [C,T,H,W]
    if clip.shape.__len__()==4:
        clip=clip.permute([1,2,3,0])
        frame = cv2.cvtColor(
            cv2.resize(((clip[clip.shape[0] // 2].detach().cpu().numpy() + 1.0) * 128.0).astype(np.uint8), origin_size),
            cv2.COLOR_BGR2RGB)

    elif clip.shape.__len__()==3:
        # [C,H,W]
        clip=clip.permute([1,2,0])
        frame = cv2.cvtColor(
            cv2.resize((clip.detach().cpu().numpy() + 128.0).astype(np.uint8), origin_size),
            cv2.COLOR_BGR2RGB)
    else:
        raise ValueError('clip expected to have [C,T,H,W] or [C,H,W] but get {}'.format(clip.shape))
    CAM=cv2.applyColorMap(CAM,cv2.COLORMAP_JET)
    result=(CAM.astype(np.float)*0.3+frame.astype(np.float)*0.5).astype(np.uint8)
    return result


## Code below is modified from https://github.com/clovaai/wsolevaluation/blob/master/evaluation.py
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))

def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W, 1)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation
    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width,_ = scoremap.shape
    scoremap_image =(scoremap * 255).astype(np.uint8)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list

class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.
    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, mask_root, multi_contour_eval):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).
        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]
            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1
        self.cnt += 1

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc

if __name__ == "__main__":
    test_CAM()
