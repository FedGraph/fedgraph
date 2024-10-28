import tenseal as ts
import pickle

def create_context():
    scheme = ts.SCHEME_TYPE.CKKS
    poly_modulus_degree = 32768  # Keep this for matrix size
    # Optimize for binary additions
    coeff_mod_bit_sizes = [60, 40, 40, 60]  # Back to simpler chain for binary
    context = ts.context(scheme, poly_modulus_degree, coeff_mod_bit_sizes)
    # Scale can be smaller for binary
    context.global_scale = 2**30  # Reduced since we're dealing with 0s and 1s
    return context

# list/array check for modulus degree

context = create_context()
secret_context = context.serialize(save_secret_key=True)

with open('fedgraph/he_context.pkl', 'wb') as f:
    pickle.dump(secret_context, f)

print("Saved HE context with secret key.")