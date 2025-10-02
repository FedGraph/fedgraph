# tests/test_threshold_ckks_min.py
import openfhe
import math

def make_cc():
    params = openfhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(openfhe.HEStd_128_classic)
    params.SetRingDim(16384)
    params.SetMultiplicativeDepth(2)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(openfhe.FLEXIBLEAUTOEXT)
    cc = openfhe.GenCryptoContext(params)
    for f in ("PKE", "SHE", "LEVELEDSHE", "MULTIPARTY"):
        cc.Enable(getattr(openfhe.PKESchemeFeature, f))
    return cc

def test_two_party_threshold_ckks_add():
    cc = make_cc()

    # Lead
    kp_lead = cc.KeyGen()
    pk0 = kp_lead.publicKey
    sk0 = kp_lead.secretKey

    # Non-lead
    kp_main = cc.MultipartyKeyGen(pk0)
    pk1 = kp_main.publicKey
    sk1 = kp_main.secretKey

    # Finalize joint PK on lead
    kp_final = cc.MultipartyKeyGen(pk1)
    joint_pk = kp_final.publicKey

    # Data
    x = [0.1, 0.2, 0.3]
    y = [0.05, 0.1, 0.15]
    scale = 2**50
    pt_x = cc.MakeCKKSPackedPlaintext(x, scale)
    pt_y = cc.MakeCKKSPackedPlaintext(y, scale)

    ct_x = cc.Encrypt(joint_pk, pt_x)
    ct_y = cc.Encrypt(joint_pk, pt_y)
    ct_sum = cc.EvalAdd(ct_x, ct_y)

    # Partial decryptions
    p_lead = cc.MultipartyDecryptLead([ct_sum], sk0)[0]
    p_main = cc.MultipartyDecryptMain([ct_sum], sk1)[0]

    fused = cc.MultipartyDecryptFusion([p_lead, p_main])
    out = fused.GetRealPackedValue()

    expect = [a+b for a,b in zip(x,y)]
    print(f"Expected: {expect}")
    print(f"Result:   {out[:len(expect)]}")
    
    assert all(abs(e-r) < 1e-3 for e,r in zip(expect, out[:len(expect)]))
    print("✅ Two-party threshold CKKS test passed!")

if __name__ == "__main__":
    test_two_party_threshold_ckks_add()


