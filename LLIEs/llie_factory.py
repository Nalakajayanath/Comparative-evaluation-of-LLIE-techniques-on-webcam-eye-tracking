from email.mime import image

from LLIEs.classical.he import histogram_equalization
from LLIEs.classical.clahe import clahe_enhancement
from LLIEs.Retinex_based.msr import multi_scale_retinex
from LLIEs.Retinex_based.ssr import single_scale_retinex
from LLIEs.classical.msrcr import msrcr
from LLIEs.deep.GAN_Based.enlightengan import EnlightenGAN
from LLIEs.deep.zeroref.zerodce import ZeroDCE
from LLIEs.deep.supervised.mirnet import MIRNet
from LLIEs.llie_enum import LLIEMethod

def apply_llie(image, method: LLIEMethod):

    if method == LLIEMethod.NONE:
        return image

    elif method == LLIEMethod.HE:
        return histogram_equalization(image)

    elif method == LLIEMethod.CLAHE:
        return clahe_enhancement(image)

    elif method == LLIEMethod.SSR:
        return single_scale_retinex(image, sigma=30)
    
    elif method == LLIEMethod.MSR:
        return multi_scale_retinex(image)
    
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