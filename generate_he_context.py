import tenseal as ts
import pickle

def create_context():
    scheme = ts.SCHEME_TYPE.CKKS
    poly_modulus_degree = 16384  #8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]  # increased
    context = ts.context(scheme, poly_modulus_degree, coeff_mod_bit_sizes)
    context.global_scale = 2**40  #2^30
    return context

context = create_context()
secret_context = context.serialize(save_secret_key=True)

with open('fedgraph/he_context.pkl', 'wb') as f:
    pickle.dump(secret_context, f)

print("Saved HE context with secret key.")