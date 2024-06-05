
import torch
import torch.nn.functional as F

from ultralytics_tools.metrics import bbox_iou, wh_iou


def smooth_label_values(eps:float=0.05, n_classes:int=2) -> tuple[float,float]:
    """
    With label smoothing, the true class label is assigned a value of
    (1 - ε), where ε is a small positive value. The remaining ε probability mass is distributed
    evenly across all other classes, giving them a value of ε/(C - 1).
    - Ref: https://arxiv.org/pdf/1902.04103
    
    Args:
        eps (float): label smoothing factor
        n_classes (int): number of classes
    Return:
        tuple[int,int]: positive, negative label smoothed targets. 
    """
    return 1.0 - eps, eps / (n_classes - 1)


class YOLOv5Loss:
    """
    YOLOv5 Loss function. Calculates class, box, and object loss for given predictions and targets.
    
    References:
        - https://docs.ultralytics.com/yolov5/tutorials/architecture_description/
        - https://github.com/ultralytics/yolov5/blob/master/utils/loss.py
    """

    def __init__(self, anchors:torch.Tensor, anchor_t:int=4, balance:list[int]=[4.0, 1.0, 0.4],
                    lambda_box=0.05, lambda_obj=0.7, lambda_cls=0.3, label_smoothing=0.0):
        """
        Args:
            - anchors (torch.Tensor): A tensor containing anchor values of shape (L, A, 2), where L is
                the number of layers, A is the number of anchors per layer, and each anchor is represented
                as (width, height).
            - anchor_t (int): Ratio threshold for matching targets to anchors. Defaults to 4.
            - balance (list[int]): A list of values to balance objectness loss contributions from each layer.
                Defaults to [4.0, 1.0, 0.4].
            - lambda_box (float): Weight for bounding box loss.
            - lambda_obj (float): Weight for objectness loss.
            - lambda_cls (float): Weight for class loss.
            - label_smoothing (float): Label smoothing value. Defaults to 0.0.
        """
        self.anchors = anchors
        self.anchor_t = anchor_t
        self.balance = balance
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.label_smoothing = label_smoothing
        # Positive, Negative target values
        self.pt_value, self.nt_value = smooth_label_values(eps=label_smoothing) 

    def __call__(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
        """
        Calculates class, box, and object loss for given predictions and targets.
        
        YOLOv5 introduces a new formulation for predicting the box coordinates in which the center point
        offset range is adjusted from (0, 1) to (-0.5, 1.5):
        - b_x = (2 * sigma(t_x) - 0.5) + c_x
        - b_y = (2 * sigma(t_y) - 0.5) + c_y 
        - b_w = p_w * (2 * sigma(t_w))^2
        - b_h = p_h * (2 * sigma(t_h))^2
        
        Due to this new formulation, instead of each target being assigned to one single cell
        (because the center point was always within the same cell), now each target can be assigned
        to multiple cells (1 to 3) due to the introduced offsets. The number of cells depends on the cell 
        location and the object center location inside the cell (See self.build_targets() for more
        information). Furthermore, each target can be assigned to multiple cell anchors.
        
        Args:
            - predictions (list[torch.Tensor]): A list of tensors representing predictions from 
                each model layer. Each tensor corresponds to predictions at different scales and 
                should have dimensions (B, A, GCj, GCi, 5+C). B is the batch size, A is the number 
                of anchors, GCj and GCi are the number of cells in the grid in each dimension (y and x) and C 
                is the number of classes.
            - targets (torch.Tensor): A tensor containing target values of shape (N, 6), where N is 
                the number of targets. Each row represents a target with format (img_id, class, x, y, w, h),
                where (x, y) represents the center coordinates of the bounding box, and (w, h) represents
                its width and height. All coordinates and size values (x, y, w, h) must be scaled in the
                range [0, 1].
                
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing a tensor with the total loss value and a 
            tensor with the box loss, object loss and class loss.
        """
        device = targets.device
        p_shape = predictions[0].shape
        
        n_batches = p_shape[0]
        n_classes = p_shape[4] - 5
        n_targets = targets.shape[0]
        
        # Initialize losses
        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        
        if n_targets > 0:
            target_classes, target_boxes, indices, target_anchors = self.build_targets(predictions, targets)

        # Calculate losses for each prediction scale
        for i, layer_pred in enumerate(predictions):
            target_obj = torch.zeros(layer_pred.shape[:4], dtype=layer_pred.dtype, device=device) # Target objectness

            if n_targets > 0 and target_classes[i].shape[0] > 0:
                n_built_targets = target_classes[i].shape[0]  # number of built targets
                img_indices, anchor_indices, cell_j, cell_i = indices[i]

                pred_xy, pred_wh, _, pred_cls = \
                    layer_pred[img_indices, anchor_indices, cell_j, cell_i].tensor_split((2, 4, 5), dim=1)
                
                # Bbox regression loss
                pred_xy = pred_xy.sigmoid() * 2 - 0.5
                pred_wh = (pred_wh.sigmoid() * 2) ** 2 * target_anchors[i]
                pred_box = torch.cat((pred_xy, pred_wh), 1)
                iou = bbox_iou(pred_box, target_boxes[i], CIoU=True).squeeze()
                box_loss += (1.0 - iou).mean()  # iou loss
                
                # Store Iou as target objectness
                iou = iou.detach().clamp(0).type(target_obj.dtype)     
                target_obj[img_indices, anchor_indices, cell_j, cell_i] = iou

                # Object type classification loss
                if n_classes > 1:  # cls loss (only if multiple classes)
                    target_cls = torch.full_like(pred_cls, self.nt_value, device=device)
                    target_cls[range(n_built_targets), target_classes[i]] = self.pt_value
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls, target_cls)

            # Objectness Loss
            pred_obj = layer_pred[..., 4]
            obj_loss += F.binary_cross_entropy_with_logits(pred_obj, target_obj) * self.balance[i]

        box_loss *= self.lambda_box
        obj_loss *= self.lambda_obj
        cls_loss *= self.lambda_cls
        
        total_loss = (box_loss + obj_loss + cls_loss) * n_batches
        loss_items = torch.cat((box_loss, obj_loss, cls_loss)).detach()
        
        return total_loss, loss_items
    
    
    def build_targets(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> tuple[list]:
        """
        Assigns targets to anchors and prepares targets for loss computation following YOLOv5
        formulation.
        
        When targets are assigned to anchors, each anchor is compared to each target and if the 
        condition is met (rmax < anchor_t), the target-anchor pair is selected. Note that the 
        same target can be assigned to different anchors in the same cell if they all meet 
        the condition.
        
        The adjacent cells to the main cell containing the center point are calculated using offsets
        that depend on the position (sector) of the center point within the current cell:
        >>>         ...
        >>>    +-----|-----+
        >>>    | l-u | r-u |
        >>> ...|-----|-----| ...
        >>>    | l-d | r-d |
        >>>    +-----|-----+
        >>>         ...
         
        Args:
            - predictions (list[torch.Tensor]): A list of tensors representing predictions from 
                each model layer. Each tensor corresponds to predictions at different scales and 
                should have dimensions (B, A, GCj, GCi, 5+C). B is the batch size, A is the number 
                of anchors, GCj and GCi are the number of cells in the grid in each dimension and C 
                is the number of classes.
            - targets (torch.Tensor): A tensor containing target values of shape (N, 6), where N is 
                the number of targets. Each row represents a target with format (img_id, class, x, y, w, h),
                where (x, y) represents the center coordinates of the bounding box, and (w, h) represents
                its width and height. All coordinates and size values (x, y, w, h) must be scaled in the
                range [0, 1].

        Returns:
            tuple: A tuple containing lists of tensors for each layer.
                - target_classes (list[Tensor]): A list of tensors, one for each layer, containing the 
                    target class labels.
                - target_boxes (list[Tensor]): A list of tensors, one for each layer, containing target 
                    ounding boxes (x, y, w, h). The (x, y) values are normalized in the range of -0.5 to 1.5,
                    while (w, h) values are adjusted with respect to the layer grid size, ranging from 0 to the
                    number of grid cells along the corresponding axis.
                - indices (list[tuple[Tensor]]): A list of tuples, one for each layer, containing tensors 
                    representing indices for sample indices, cell anchor indices, and grid cell indices. 
                    They are used to extract the corresponding model predictions, in which a ground truth 
                    object has been assigned.
                - target_anchors (list[Tensor]): A list of tensors, one for each layer, containing the anchor
                    (w, h) values of each selected cell anchor.
        """
        device = targets.device
        p_shape = predictions[0].shape

        n_anchors = p_shape[1]
        n_targets = targets.shape[0]
        n_layers = len(predictions)

        if n_targets == 0:
            raise ValueError("No targets provided")
        if len(predictions) != self.anchors.shape[0]:
            raise ValueError("Number of layers and anchors do not match")
        if n_anchors != self.anchors.shape[1]:
            raise ValueError("Number of layer bounding boxes and provided anchors do not match")
        
        target_classes, target_boxes, indices, target_anchors = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain

        anchor_indices = torch.arange(n_anchors, device=device).float()
        anchor_indices = anchor_indices.view(n_anchors, 1).repeat(1, n_targets)
        # Append anchor indices at the end of each target
        target_anchor_pairs = torch.cat((targets.repeat(n_anchors, 1, 1), anchor_indices[..., None]), dim=2)

        g = 0.5 # bias
        # Define offsets for each direction in the grid cell. Offsets will be substracted 
        # from grid cell center point to select adjacent cells.
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * g
        # -> [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1] -> current, left, up, right, down
        
        assigned_targets = torch.zeros((n_targets,), device=device)
        for i in range(n_layers):
            layer_anchors, layer_shape = self.anchors[i], predictions[i].shape
            # Adjust scale of targets based on layer grid size
            gain[2:6] = torch.tensor(layer_shape)[[3, 2, 3, 2]]  # xyxy gain
            scaled_ta_pairs = target_anchor_pairs * gain

            # Match targets to anchors
            ratio = scaled_ta_pairs[..., 4:6] / layer_anchors[:, None]  # wh ratio
            rmax = torch.max(ratio, 1 / ratio).max(dim=2)[0]
            mask_ = rmax < self.anchor_t
            selected_ta_pairs = scaled_ta_pairs[mask_]  # filter
            assigned_targets += mask_.sum(dim=0)

            # Compute offsets
            gxy = selected_ta_pairs[:, 2:4]  # grid xy (x increases right, y increases down)
            gxy_inv = gain[[2, 3]] - gxy   # grid xy inverse (x increases left, y increases up)
            # -- Check if center point is in the left | up sectors of the grid cell
            left, up = ((gxy % 1 < g) & (gxy > 1)).T
            # -- Check if center point is in the right | down sectors of the grid cell
            right, down = ((gxy_inv % 1 < g) & (gxy_inv > 1)).T
            # -- Stack boolean masks and compute offsets
            current = torch.ones_like(left) # All true. To select the cells in which the center point is located
            stacked_conditions = torch.stack((current, left, up, right, down))
            built_targets = selected_ta_pairs.repeat((5, 1, 1))[stacked_conditions]
            built_target_offsets = (torch.zeros_like(gxy)[None] + off[:, None])[stacked_conditions]            

            # Prepare built-targets for loss computation 
            bc, gxy, gwh, a = built_targets.chunk(4, 1)
            anchor_indices = a.long().view(-1)
            img_indices, classes = bc.long().T
            # -- Compute grid cell indices
            gij = (gxy - built_target_offsets).long()
            gi, gj = gij.T
            gj = gj.clamp(0, layer_shape[2] - 1)
            gi = gi.clamp(0, layer_shape[3] - 1)
            # -- Append results
            indices.append((img_indices, anchor_indices, gj, gi))
            target_boxes.append(torch.cat((gxy - gij, gwh), dim=1))
            target_anchors.append(layer_anchors[anchor_indices])
            target_classes.append(classes) 
        
        msg = (f"There are targets not assigned to any anchor in any layer: {assigned_targets}. " + 
                "Try to increase the 'anchor_t' value or use better anchors")
        assert (assigned_targets > 0).all(), msg    

        return target_classes, target_boxes, indices, target_anchors

class YOLOv3Loss:
    """ 
    YOLOv3 Loss function. Calculates class, box, and object loss for given predictions and targets.
    
    References:
        - https://arxiv.org/pdf/1804.02767
        - https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
    """

    def __init__(self, anchors:torch.Tensor, iou_t=0.3, ignore_threshold=0.5,
                    lambda_box=1, lambda_obj=1, lambda_noobj=0.5, lambda_cls=1,
                        label_smoothing=0.0):
        """ 
        Args:
            - anchors (torch.Tensor): A tensor containing anchor values of shape (L, A, 2), where L is
                the number of layers, A is the number of anchors per layer, and each anchor is represented
                as (width, height).
            - iou_t (float): Minimum IoU threshold for assigning targets to anchors.
            - ignore_threshold (float): IoU threshold for ignoring loss computation in anchors of the target
                cell that have not been selected as the best one but have a high IoU with the object.
            - lambda_box (float): Weight for bounding box loss.
            - lambda_obj (float): Weight for objectness loss.
            - lambda_noobj (float): Weight for non-objectness loss.
            - lambda_cls (float): Weight for class loss.
            - label_smoothing (float): Label smoothing value. Defaults to 0.0.
        """
        self.anchors = anchors
        self.iou_t = iou_t
        self.ignore_threshold = ignore_threshold
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.label_smoothing = label_smoothing
        # Positive, Negative target values
        self.pt_value, self.nt_value = smooth_label_values(eps=label_smoothing) 

    def __call__(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> 'torch.Tensor | tuple':
        """
        Calculates class, box, and object loss for given predictions and targets.
        
        YOLOv3 formulation for predicting box coordinates:
        - b_x = sigma(t_x) + c_x
        - b_y = sigma(t_y) + c_y 
        - b_w = p_w * exp(t_w)
        - b_h = p_h * exp(t_h)
        
        Because of this formulation the object center point is always within the same cell, so each target is
        only assigned to one cell (different from YOLOv5 approach). Additionally, each target is assigned
        only to the best anchor box of the cell (See self.build_targets() for more information).
        
        Args:
            - predictions (list[torch.Tensor]): A list of tensors representing predictions from 
                each model layer. Each tensor corresponds to predictions at different scales and 
                should have dimensions (B, A, GCj, GCi, 5+C). B is the batch size, A is the number 
                of anchors, GCj and GCi are the number of cells in the grid in each dimension and C 
                is the number of classes.
            - targets (torch.Tensor): A tensor containing target values of shape (N, 6), where N is 
                the number of targets. Each row represents a target with format (img_id, class, x, y, w, h),
                where (x, y) represents the center coordinates of the bounding box, and (w, h) represents
                its width and height. All coordinates and size values (x, y, w, h) must be scaled in the 
                range [0, 1].
                
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing a tensor with the total loss value and a 
            tensor with the box loss, object loss, noobj_loss and class loss.
        """
        device = targets.device
        p_shape = predictions[0].shape
        
        n_batches = p_shape[0]
        n_classes = p_shape[4] - 5
        n_targets = targets.shape[0]
        
        # Initialize losses
        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        noobj_loss = torch.zeros(1, device=device)
        
        # Build targets for each prediction scale
        if n_targets > 0:
            btargets = self.build_targets(predictions, targets)
            target_classes, target_boxes, obj_indices, target_anchors, noobj_indices = btargets
            
        # Calculate losses for each prediction scale
        for i, layer_pred in enumerate(predictions):

            if n_targets > 0 and target_classes[i].shape[0] > 0:
                n_built_targets = target_classes[i].shape[0]  # number of built targets
                
                imgi, anchi, gj, gi = obj_indices[i]
                pred_xy, pred_wh, pred_obj, pred_cls = \
                    layer_pred[imgi, anchi, gj,gi].tensor_split((2, 4, 5), dim=1)
                
                # Bbox regression loss
                pred_xy = pred_xy.sigmoid()
                pred_wh = pred_wh.exp() * target_anchors[i]
                box_loss += F.mse_loss(pred_xy, target_boxes[i][...,:2]).squeeze()
                box_loss += F.mse_loss(pred_wh.sqrt(), target_boxes[i][..., 2:].sqrt()).squeeze()
                
                # Objectness loss
                target_obj = torch.ones_like(pred_obj, device=device)
                obj_loss += F.binary_cross_entropy_with_logits(pred_obj, target_obj)
                
                # Object type classification loss
                if n_classes > 1:  # cls loss (only if multiple classes)
                    target_cls = torch.full_like(pred_cls, self.nt_value, device=device)
                    target_cls[range(n_built_targets), target_classes[i]] = self.pt_value
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls, target_cls)

            # No-object loss
            imgi, anchi, gj, gi = noobj_indices[i]
            pred_noobj = layer_pred[imgi, anchi, gj, gi][..., 4]
            target_noobj = torch.zeros_like(pred_noobj, device=device)
            noobj_loss += F.binary_cross_entropy_with_logits(pred_noobj, target_noobj)

        box_loss *= self.lambda_box
        obj_loss *= self.lambda_obj
        noobj_loss *= self.lambda_noobj
        cls_loss *= self.lambda_cls
        
        total_loss = (box_loss + obj_loss + noobj_loss + cls_loss) * n_batches
        loss_items = torch.cat((box_loss, obj_loss, noobj_loss, cls_loss)).detach()
        
        return total_loss, loss_items
    
    
    def build_targets(self, predictions:list[torch.Tensor], targets:torch.Tensor) -> tuple[list]:
        """
        Assigns targets to anchors and prepares targets for loss computation following YOLOv3
        formulation.
        
        When targets are assigned to anchors, each target is compared to each anchor computing the
        IoU metric. The anchor with the highest IoU will be assigned to the target if it is above a 
        defined threshold (iou_t). Therefore, each target can only be assigned to one anchor of the
        cell that contains the center point of the object in that specific prediction layer.
        However, the same target can be assigned at the same time to other cell anchor of different 
        prediction layers.
         
        Args:
            - predictions (list[torch.Tensor]): A list of tensors representing predictions from 
                each model layer. Each tensor corresponds to predictions at different scales and 
                should have dimensions (B, A, GCj, GCi, 5+C). B is the batch size, A is the number 
                of anchors, GCj and GCi are the number of cells in the grid in each dimension and C 
                is the number of classes.
            - targets (torch.Tensor): A tensor containing target values of shape (N, 6), where N is 
                the number of targets. Each row represents a target with format (img_id, class, x, y, w, h),
                where (x, y) represents the center coordinates of the bounding box, and (w, h) represents
                its width and height. All coordinates and size values (x, y, w, h) must be scaled in the
                range [0, 1].

        Returns:
            tuple: A tuple containing lists of tensors for each layer.
                - target_classes (list[Tensor]): A list of tensors, one for each layer, containing the 
                    target class labels.
                - target_boxes (list[Tensor]): A list of tensors, one for each layer, containing target 
                    ounding boxes (x, y, w, h). The (x, y) values are normalized in the range of -0.5 to 1.5,
                    while (w, h) values are adjusted with respect to the layer grid size, ranging from 0 to the
                    number of grid cells along the corresponding axis.
                - indices (list[tuple[Tensor]]): A list of tuples, one for each layer, containing tensors 
                    representing indices for sample indices, cell anchor indices, and grid cell indices. 
                    They are used to extract the corresponding model predictions, in which a ground truth 
                    object has been assigned.
                - target_anchors (list[Tensor]): A list of tensors, one for each layer, containing the anchor
                    (w, h) values of each selected cell anchor.
                - noobj_indices (list[tuple[Tensor]]): A list of tuples, one for each layer, containing tensors 
                    representing indices for sample indices, cell anchor indices, and grid cell indices. 
                    They are used to extract the corresponding model predictions, in which no ground truth 
                    object has been assigned.
        """
        device = targets.device
        p_shape = predictions[0].shape

        n_anchors = p_shape[1]
        n_targets = targets.shape[0]
        n_layers = len(predictions)

        if n_targets == 0:
            raise ValueError("No targets provided")
        if len(predictions) != self.anchors.shape[0]:
            raise ValueError("Number of layers and anchors do not match")
        if n_anchors != self.anchors.shape[1]:
            raise ValueError("Number of layer bounding boxes and provided anchors do not match")
        
        target_classes, target_boxes, obj_indices, target_anchors, noobj_indices = [], [], [], [], []
        gain = torch.ones(6, device=device)  # normalized to gridspace gain

        assigned_targets = torch.zeros((n_targets,), device=device)
        for i in range(n_layers):
            layer_anchors, layer_shape = self.anchors[i], predictions[i].shape
            # Define masks 
            obj_mask = torch.zeros(layer_shape[:4], dtype=torch.bool, device=device)
            noobj_mask = torch.ones(layer_shape[:4], dtype=torch.bool, device=device)
            # Adjust scale of targets based on layer grid size
            gain[2:6] = torch.tensor(layer_shape)[[3, 2, 3, 2]]  # xyxy gain
            scaled_targets = targets * gain
            
            # Match targets to best anchors based on IoU
            ious = wh_iou(layer_anchors, scaled_targets[..., 4:6])
            best_ious, anchor_indices_ = ious.max(0)
            mask_ = best_ious > self.iou_t
            assigned_targets += mask_.long()
            layer_matched_targets = scaled_targets[mask_]
            
            # Set noobj mask to zero where target-anchor IoU in selected cells exceeds ignore threshold
            img_indices_ = scaled_targets[:, :1].long()
            gi_, gj_ = scaled_targets[:, 2:4].long().t()
            for i, anchor_ious in enumerate(ious.t()):
                noobj_mask[img_indices_[i], anchor_ious > self.ignore_threshold, gj_[i], gi_[i]] = 0
            
            # Prepare targets for loss computation
            best_anchor_indices = anchor_indices_[mask_].long()
            bc, gxy, gwh = layer_matched_targets.chunk(3, 1)
            img_indices, classes = bc.long().T
            gij = gxy.long()
            gi, gj = gij.T
            gj = gj.clamp(0, layer_shape[2] - 1)
            gi = gi.clamp(0, layer_shape[3] - 1)
            # -- Update masks
            obj_mask[img_indices, best_anchor_indices, gj, gi] = 1
            noobj_mask[img_indices, best_anchor_indices, gj, gi] = 0
            # -- Append results
            target_boxes.append(torch.cat((gxy - gij, gwh), dim=1))
            target_anchors.append(layer_anchors[best_anchor_indices])
            target_classes.append(classes)
            # -- Note: Independent indices are more efficient for indexing than boolean masks 
            # -> time(mask.nonzero()) + time(t[ind1,ind2,...]) < time(t[mask])
            obj_ind = (img_indices, best_anchor_indices, gj, gi)
            obj_indices.append(obj_ind)      # Indices to extract predictions tasked with predicting an object
            noobj_ind = torch.nonzero(noobj_mask, as_tuple=True)
            noobj_indices.append(noobj_ind)  # Indices to extract predictions where no obj has been assigned
        
        msg = (f"There are targets not assigned to any anchor in any layer: {assigned_targets}. " + 
                "Try to lower the 'iou_t' value or use better anchors")
        assert (assigned_targets > 0).all(), msg
            
        return target_classes, target_boxes, obj_indices, target_anchors, noobj_indices