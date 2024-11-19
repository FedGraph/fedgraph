import pickle

# import tenseal as ts


def create_training_context():
    scheme = ts.SCHEME_TYPE.CKKS

    # Keep the same settings that worked for features
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]

    context = ts.context(
        scheme=scheme,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )

    # Higher scale for better precision with small parameter values
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.auto_relin = True
    context.auto_rescale = True

    return context


if __name__ == "__main__":
    training_context = create_training_context()
    training_secret_context = training_context.serialize(save_secret_key=True)

    with open("fedgraph/he_context.pkl", "wb") as f:
        pickle.dump(training_secret_context, f)
    print("Saved HE context with secret key.")
