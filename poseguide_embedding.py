import math
import numpy as np
def pose_embedding_module(model,
                    img,
                    result,
                    img_w,
                    img_h,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector of mmpose.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        img_w (int): wide of image
        img_h (int): wide of image
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default False.
        out_file (str|None): The filename of the output visualization image.
    """
    radius = max(int(math.sqrt(img_w*img_h)/30),3)

    thickness = max(int(math.sqrt(img_w*img_h)/90),1)
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
    palette_kpt = np.array([[0,255,0],[255,0,0],[0,0,255],[128,0,0],[0,0,128],[255,0,255],[0,255,255],[255,255,0],[0,128,128],[128,128,0]])

    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                   'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette_kpt[[
           0, 0, 0, 0, 0, 3, 4, 3 , 4, 1, 2, 5, 5, 6 , 7, 8, 9
        ]]


    else:
        raise NotImplementedError()

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img