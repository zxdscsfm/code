import runpy
import sys

import torch


def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
    return self


def _module_cuda(self, device=None):
    return self


torch.Tensor.cuda = _tensor_cuda
torch.nn.Module.cuda = _module_cuda

sys.argv = [
    "infer_fedlppa_personalized_tmp_v1dg2.py",
    "--root_path",
    "../data/ODOC_h5",
    "--exp",
    "odoc/FedLPPA_v1-dg2_odoc_paper_r500_l10_odoc_20260503_171001_seed2022",
    "--checkpoint_kind",
    "best",
    "--output_dir",
    "../model/odoc/FedLPPA_v1-dg2_odoc_paper_r500_l10_odoc_20260503_171001_seed2022/test_best_formal_cpu",
    "--method_name",
    "ODOC_v1-dg2_best",
    "--model",
    "unet_univ5",
    "--img_class",
    "odoc",
    "--num_classes",
    "3",
    "--in_chns",
    "3",
    "--img_size",
    "384",
    "--min_num_clients",
    "5",
    "--prompt",
    "universal",
    "--attention",
    "dual",
    "--dual_init",
    "aggregated",
    "--label_prompt",
    "1",
    "--geometry_guided",
    "1",
    "--gpu",
    "0",
    "--site_labels",
    "SiteA",
    "SiteB",
    "SiteC",
    "SiteD",
    "SiteE",
    "--clients",
    "client1",
    "client2",
    "client3",
    "client4",
    "client5",
    "--sup_type_list",
    "scribble",
    "scribble_noisy",
    "scribble_noisy",
    "keypoint",
    "block",
]

runpy.run_path("infer_fedlppa_personalized_tmp_v1dg2.py", run_name="__main__")
