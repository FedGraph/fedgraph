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
    for feature in (
        openfhe.PKE,
        openfhe.KEYSWITCH,
        openfhe.LEVELEDSHE,
        openfhe.ADVANCEDSHE,
        openfhe.MULTIPARTY,
    ):
        cc.Enable(feature)
    return cc

def test_two_party_threshold_ckks_add():
    cc = make_cc()

    # Lead
    kp_lead = cc.KeyGen()
    pk0 = kp_lead.publicKey
    sk0 = kp_lead.secretKey

    # Non-lead. Its public key is the joint public key.
    kp_main = cc.MultipartyKeyGen(pk0)
    joint_pk = kp_main.publicKey
    sk1 = kp_main.secretKey

    # Data
    x = [0.1, 0.2, 0.3]
    y = [0.05, 0.1, 0.15]
    pt_x = cc.MakeCKKSPackedPlaintext(x)
    pt_y = cc.MakeCKKSPackedPlaintext(y)
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
