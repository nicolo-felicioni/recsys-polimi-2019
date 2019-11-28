from Base.Evaluation import MyEvaluator


def evaluate(data, random_seed_used_in_data, rec, rec_name, rec_args, filename="logToFileEvaluator.csv"):
    f = open(filename, "a+")
    f.write("random_seed, rec_name, rec_args, type_of_users, map\n")
    print("random_seed, rec_name, rec_args, type_of_users, map\n")
    for n, users, description in data.urm_train_users_by_type:
        eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, users, rec, at=10, remove_top=0)
        print(f"{random_seed_used_in_data}, {rec_name}, {rec_args}, {description}, {map}\n")
        f.write(f"{random_seed_used_in_data}, {rec_name}, {rec_args}, {description}, {map}\n")
        f.flush()
    eval, map = MyEvaluator.evaluate_algorithm(data.urm_test, data.ids_target_users, rec, at=10,
                                               remove_top=0)
    print(f"{random_seed_used_in_data}, {rec_name}, {rec_args}, target_users, {map}\n")
    f.write(f"{random_seed_used_in_data}, {rec_name}, {rec_args}, target_users, {map}\n")
    f.flush()