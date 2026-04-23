from email.mime import image

from LLIEs.classical.he import histogram_equalization
from LLIEs.classical.clahe import clahe_enhancement
from LLIEs.Retinex_based.msr import multi_scale_retinex
from LLIEs.Retinex_based.ssr import single_scale_retinex
from LLIEs.Retinex_based.msrcr import msrcr
from LLIEs.deep.GAN_Based.enlightengan import EnlightenGAN
from LLIEs.deep.zeroref.zerodce import ZeroDCE
from LLIEs.deep.supervised.mirnet import MIRNet
from LLIEs.llie_enum import LLIEMethod

def apply_llie(image, method: LLIEMethod, params=None):

    if params is None:
        params = {}

    if method == LLIEMethod.NONE:
        return image

    elif method == LLIEMethod.HE:
        return histogram_equalization(image)

    elif method == LLIEMethod.CLAHE:
        clip = params.get("clip_limit", 2.0)
        return clahe_enhancement(image, clip_limit=clip)

    elif method == LLIEMethod.SSR:
        sigma = params.get("sigma", 30)
        return single_scale_retinex(image, sigma=sigma)
    
    elif method == LLIEMethod.MSR:
        sigmas = params.get("sigmas", [15, 80, 250])
        return multi_scale_retinex(image, sigmas=sigmas)
    
    elif method == LLIEMethod.MSRCR:
        return msrcr(image)
    
    elif method == LLIEMethod.ZERODCE:
        if not hasattr(apply_llie, "zerodce_model"):
            apply_llie.zerodce_model = ZeroDCE()
        return apply_llie.zerodce_model.enhance(image)

    elif method == LLIEMethod.ENLIGHTENGAN:
        if not hasattr(apply_llie, "enlightengan_model"):
            apply_llie.enlightengan_model = EnlightenGAN()
        return apply_llie.enlightengan_model.enhance(image)
    
    elif method == LLIEMethod.MIRNET:
        if not hasattr(apply_llie, "mirnet_model"):
            apply_llie.mirnet_model = MIRNet()
        return apply_llie.mirnet_model.enhance(image)

    else:
        raise ValueError("Unsupported LLIE method")