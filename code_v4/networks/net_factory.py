from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_320, UNet_DS, UNet_CCT, UNet_CCT_3H, UNet_Head, UNet_MultiHead, \
                            UNet_LC, UNet_LC_MultiHead,UNet_LC_MultiHead_Two, UNet_Uni, UNet_UniV2, UNet_UniV3, UNet_UniV4, UNet_UniV5, UNet_Univ5_Ablation, UNet_UniV5_WO_Uni_Prompt, UNet_UniV5_AttentionConcat, UNet_LC_Auxi

# from utils.TreeEnergyLoss.lib.models.nets.fcnet import FcnNet
# from utils.TreeEnergyLoss.lib.models.nets.treefcn import TreeFCN
# from utils.TreeEnergyLoss.lib.models.nets.deeplabv3plus import DeepLabV3Plus
# from utils.TreeEnergyLoss.lib.utils.tools.configer import Configer


def net_factory(args, net_type="unet", in_chns=1, class_num=3):
    device = getattr(args, "device", None)
    if device is None:
        device = "cuda" if getattr(args, "use_cuda", 1) == 1 else "cpu"
    def _to_device(model):
        return model.to(device)
    if net_type == "unet":
        net = _to_device(UNet(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_320":
        net = _to_device(UNet_320(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_cct":
        net = _to_device(UNet_CCT(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_cct_3h":
        net = _to_device(UNet_CCT_3H(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_ds":
        net = _to_device(UNet_DS(in_chns=in_chns, class_num=class_num))
    elif net_type == "efficient_unet":
        net = _to_device(Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num))
    elif net_type == "pnet":
        net = _to_device(PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]))
    elif net_type == "unet_head":
        net = _to_device(UNet_Head(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_multihead":
        net = _to_device(UNet_MultiHead(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_lc":
        net = _to_device(UNet_LC(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_lc_auxi":
        net = _to_device(UNet_LC_Auxi(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_lc_multihead":
        net = _to_device(UNet_LC_MultiHead(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_lc_multihead_two":
        net = _to_device(UNet_LC_MultiHead_Two(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_uni":
        net = _to_device(UNet_Uni(in_chns=in_chns, class_num=class_num, client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_univ2":
        net = _to_device(UNet_UniV2(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid))
    elif net_type == "unet_univ3":
        net = _to_device(UNet_UniV3(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size))
    elif net_type == "unet_univ4":
        net = _to_device(UNet_UniV4(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size))

    elif net_type == "unet_univ5":
        net = _to_device(UNet_UniV5(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size))
    elif net_type == "unet_univ5_ablation":
        net = _to_device(UNet_Univ5_Ablation(in_chns=in_chns, class_num=class_num))
    elif net_type == "unet_univ5_wo_uniprompt":
        net = _to_device(UNet_UniV5_WO_Uni_Prompt(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size))
    elif net_type == "unet_univ5_attention_concat":
        net = _to_device(UNet_UniV5_AttentionConcat(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size))
    
    else:
        net = None
    return net
