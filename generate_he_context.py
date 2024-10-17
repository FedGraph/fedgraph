import tenseal as ts
import pickle

def create_context():
    scheme = ts.SCHEME_TYPE.CKKS
    poly_modulus_degree = 16384  # Increased from 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]  # Increased bit sizes
    context = ts.context(scheme, poly_modulus_degree, coeff_mod_bit_sizes)
    context.global_scale = 2**40  # Increased from 2^30
    return context

# Create and save the context with secret key
context = create_context()
secret_context = context.serialize(save_secret_key=True)

with open('fedgraph/he_context.pkl', 'wb') as f:
    pickle.dump(secret_context, f)

print("Saved HE context with secret key.")# import tenseal as ts
# import pickle

# def create_he_context():
#     context = ts.context(
#         ts.SCHEME_TYPE.CKKS,
#         poly_modulus_degree=8192,
#         coeff_mod_bit_sizes=[60, 40, 40, 60]
#     )
#     context.global_scale = 2**40
#     context.generate_galois_keys()
#     return context

# def save_context(context, filename='he_context.pkl'):
#     with open(filename, 'wb') as f:
#         pickle.dump(context.serialize(), f)

# def load_context(filename='he_context.pkl'):
#     with open(filename, 'rb') as f:
#         context_bytes = pickle.load(f)
#     return ts.context_from(context_bytes)

# # Create and save the context (do this once, before creating Server and Trainers)
# he_context = create_he_context()
# save_context(he_context)

# # In Server and Trainer classes, replace the setup_he_context method with:
# def setup_he_context(self):
#     return load_context()