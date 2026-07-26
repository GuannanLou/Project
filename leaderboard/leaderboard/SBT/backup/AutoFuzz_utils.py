from argparse import Namespace
import numpy as np
import os
def parse_fuzzing_arguments():
    # [ego_linear_speed, min_d, d_angle_norm, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision]
    default_objective_weights = np.array([-1., 1., 1., 1., 1., -1., 0., 0., 0., 0.])
    default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])
    default_check_unique_coeff = [0, 0.1, 0.5]
    WARMUP_PATH = '<path-to-warm-up-run-folder>'

    fuzzing_arguments = Namespace(
        # general
        route_type='town07_front_0',
        scenario_type='go_straight_town07',
        ego_car_model='lbc',  # 默认值，未在命令行指定
        algorithm_name='nsga2-un',
        
        ports=[2021],  # -p 2021
        scheduler_port=8795,  # -s 8795
        dashboard_address=8796,  # -d 8796
        
        simulator='carla',  # 默认值，未在命令行指定
        random_seed=0,  # 默认值

        # carla specific
        has_display='0',  # --has_display 0
        debug=1,  # 默认值
        correct_spawn_locations_after_run=0,  # 默认值
        
        # carla_op specific
        carla_path="../carla_0911_rss/CarlaUE4.sh",  # 默认值

        # no_simulation specific
        no_simulation_data_path=None,  # 默认值
        objective_labels=[],  # 默认值
        
        # logistic
        root_folder='carla_lbc/run_results',  # 默认值
        parent_folder='',  # 默认值
        mean_objectives_across_generations_path='',  # 默认值
        episode_max_time=60,  # 默认值
        record_every_n_step=5,  # --record_every_n_step 5
        gpus='0,1',  # 默认值

        # algorithm related
        n_gen=15,  # --n_gen 15
        pop_size=50,  # --pop_size 50
        survival_multiplier=1,  # 默认值
        n_offsprings=300,  # 默认值
        has_run_num=700,  # --has_run_num 700
        sample_multiplier=200,  # 默认值
        mating_max_iterations=200,  # 默认值
        only_run_unique_cases=1,  # --only_run_unique_cases 1
        consider_interested_bugs=1,  # 默认值

        outer_iterations=3,  # 默认值
        objective_weights=np.array([-1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]),  # --objective_weights -1 1 1 0 0 0 0 0 0 0
        default_objectives=default_objectives,  # 默认值
        standardize_objective=0,  # 默认值
        normalize_objective=0,  # 默认值
        traj_dist_metric='nearest',  # 默认值

        check_unique_coeff=[0, 0.1, 0.5],  # --check_unique_coeff 0 0.1 0.5
        use_single_objective=1,  # 默认值
        rank_mode='adv_nn',  # --rank_mode adv_nn
        ranking_model='nn_pytorch',  # 默认值
        initial_fit_th=100,  # 默认值
        min_bug_num_to_fit_dnn=10,  # 默认值

        pgd_eps=1.01,  # 默认值
        adv_conf_th=-4,  # 默认值
        attack_stop_conf=0.9,  # 默认值
        use_single_nn=1,  # 默认值

        warm_up_path=WARMUP_PATH,  # --warm_up_path <path-to-warm-up-run-folder>
        warm_up_len=500,  # --warm_up_len 500
        regression_nn_use_running_data=1,  # 默认值

        sample_avoid_ego_position=0,  # 默认值
        
        uncertainty='',  # 默认值
        model_type='one_output',  # 默认值
        
        termination_condition='generations',  # 默认值
        max_running_time=3600*24,  # 默认值
        
        emcmc=0,  # 默认值
        use_unique_bugs=1,  # 默认值
        finish_after_has_run=1  # 默认值
    )

    os.environ['HAS_DISPLAY'] = fuzzing_arguments.has_display
    os.environ['CUDA_VISIBLE_DEVICES'] = fuzzing_arguments.gpus
    fuzzing_arguments.objective_weights = np.array(fuzzing_arguments.objective_weights)
    # ['BNN', 'one_output']
    # BALD and BatchBALD only support BNN
    if fuzzing_arguments.uncertainty.split('_')[0] in ['BALD', 'BatchBALD']:
        fuzzing_arguments.model_type = 'BNN'

    if 'un' in fuzzing_arguments.algorithm_name:
        fuzzing_arguments.use_unique_bugs = 1
    else:
        fuzzing_arguments.use_unique_bugs = 0

    if fuzzing_arguments.algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
        fuzzing_arguments.emcmc = 1
    else:
        fuzzing_arguments.emcmc = 0

    return fuzzing_arguments



# ---------------- ADV -------------------
def if_violate_constraints_vectorized(X, customized_constraints, labels, ego_start_position=None, verbose=False):
    labels_to_id = {label: i for i, label in enumerate(labels)}

    keywords = ["coefficients", "labels", "value"]
    extra_keywords = ["power"]

    if_violate = False
    violated_constraints = []
    involved_labels = set()

    X = np.array(X)
    remaining_inds = np.arange(X.shape[0])

    for i, constraint in enumerate(customized_constraints):
        for k in keywords:
            assert k in constraint
        assert len(constraint["coefficients"]) == len(constraint["labels"])

        ids = np.array([labels_to_id[label] for label in constraint["labels"]])


        # x_ids = [x[id] for id in ids]
        if "powers" in constraint:
            powers = np.array(constraint["powers"])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint["coefficients"])

        if_violate_current = (
            np.sum(coeff * np.power(X[remaining_inds[:, None], ids], powers), axis=1) > constraint["value"]
        )
        remaining_inds = remaining_inds[if_violate_current==0]

    # beta: eliminate NPC vehicles having generation collision with the ego car
    # TBD: consider customized_center_transforms, customizable NPC vehicle size
    # also only consider OP for now
    print('remaining_inds before', len(remaining_inds))
    tmp_remaining_inds = remaining_inds.copy()
    if ego_start_position:
        j = 0
        ego_x, ego_y, ego_yaw = ego_start_position
        ego_w = 0.93
        vehicle_w_j = 0.93
        ego_l = 2.35
        vehicle_l_j = 2.35
        dw = ego_w + vehicle_w_j
        dl = ego_l + vehicle_l_j
        while 'vehicle_x_'+str(j) in labels:
            remaining_inds_i = remaining_inds.copy()

            x_ind = labels.index('vehicle_x_'+str(j))
            y_ind = labels.index('vehicle_y_'+str(j))

            vehicle_x_j = X[remaining_inds_i, x_ind]
            vehicle_y_j = X[remaining_inds_i, y_ind]

            dx_rel = vehicle_x_j
            dy_rel = vehicle_y_j


            x_far_inds = remaining_inds_i[np.abs(dx_rel) > dw]
            x_close_inds = remaining_inds_i[np.abs(dx_rel) <= dw]

            y_far_inds = x_close_inds[np.abs(dy_rel[x_close_inds]) > dl]

            remaining_inds_i = np.concatenate([x_far_inds, y_far_inds])
            tmp_remaining_inds = np.intersect1d(tmp_remaining_inds, remaining_inds_i)
            j += 1
    remaining_inds = tmp_remaining_inds


    if verbose:
        print('constraints filtering', len(X), '->', len(remaining_inds))

    return remaining_inds

def customized_standardize(X, standardize, m, partial=True, scale_only=False):
    # print(X[:, :m].shape, standardize.transform(X[:, m:]).shape)
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.transform(X[:, m:])
        res = np.concatenate([X[:, :m], standardize.transform(X[:, m:])], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.transform(X)
    return res

def customized_inverse_standardize(X, standardize, m, partial=True, scale_only=False):
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.inverse_transform(X[:, m:])
        res = np.concatenate([X[:, :m], res_non_encode], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.inverse_transform(X)
    return res

def recover_fields_not_changing(x, x_removed, kept_fields, removed_fields):
    n = x.shape[0]
    m = len(kept_fields) + len(removed_fields)

    # this is True usually when adv is used
    if x_removed.shape[0] != n:
        x_removed = np.array([x_removed[0] for _ in range(n)])
    x_recovered = np.zeros([n, m])
    x_recovered[:, kept_fields] = x
    x_recovered[:, removed_fields] = x_removed

    return x_recovered

def decode_fields(x, enc, inds_to_encode, inds_non_encode, encode_fields, adv=False):
    n = x.shape[0]
    m = len(inds_to_encode) + len(inds_non_encode)
    embed_dims = np.sum(encode_fields)

    embed = x[:, :embed_dims]
    kept = x[:, embed_dims:]

    if adv:
        one_hot_embed = np.zeros(embed.shape)
        s = 0
        for field_len in encode_fields:
            max_inds = np.argmax(x[:, s : s + field_len], axis=1)
            one_hot_embed[np.arange(x.shape[0]), s + max_inds] = 1
            s += field_len
        embed = one_hot_embed

    x_encoded = enc.inverse_transform(embed)
    # print('encode_fields', encode_fields)
    # print('embed', embed[0], x_encoded[0])
    x_decoded = np.zeros([n, m])
    x_decoded[:, inds_non_encode] = kept
    x_decoded[:, inds_to_encode] = x_encoded

    return x_decoded

# ---------------- Uniqueness -------------------
def is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=True):
    if len(cur_X) == 0:
        return []
    cur_X = np.array(cur_X)
    prev_X = np.array(prev_X)
    eps = 1e-10
    remaining_inds = np.arange(cur_X.shape[0])

    mask = np.array(mask)
    xl = np.array(xl)
    xu = np.array(xu)

    n = len(mask)

    variant_fields = (xu - xl) > eps
    variant_fields_num = np.sum(variant_fields)
    th_num = np.max([np.round(th * variant_fields_num), 1])

    mask = mask[variant_fields]
    int_inds = mask == "int"
    real_inds = mask == "real"
    xl = xl[variant_fields]
    xu = xu[variant_fields]
    xl = np.concatenate([np.zeros(np.sum(int_inds)), xl[real_inds]])
    xu = np.concatenate([0.99*np.ones(np.sum(int_inds)), xu[real_inds]])

    # hack: backward compatibility with previous run data
    # if cur_X.shape[1] == n-1:
    #     cur_X = np.concatenate([cur_X, np.zeros((cur_X.shape[0], 1))], axis=1)

    cur_X = cur_X[:, variant_fields]
    cur_X = np.concatenate([cur_X[:, int_inds], cur_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

    if len(prev_X) > 0:
        prev_X = prev_X[:, variant_fields]
        prev_X = np.concatenate([prev_X[:, int_inds], prev_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

        diff_raw = np.abs(np.expand_dims(cur_X, axis=1) - np.expand_dims(prev_X, axis=0))
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=2)
        equal = diff_norm < th_num
        remaining_inds = np.mean(equal, axis=1) == 0
        remaining_inds = np.arange(cur_X.shape[0])[remaining_inds]

        # print('remaining_inds', remaining_inds, np.arange(cur_X.shape[0])[remaining_inds], cur_X[np.arange(cur_X.shape[0])[remaining_inds]])
        if verbose:
            print('prev X filtering:',cur_X.shape[0], '->', len(remaining_inds))

    if len(remaining_inds) == 0:
        return []

    cur_X_remaining = cur_X[remaining_inds]
    print('len(cur_X_remaining)', len(cur_X_remaining))
    unique_inds = []
    for i in range(len(cur_X_remaining)-1):
        diff_raw = np.abs(np.expand_dims(cur_X_remaining[i], axis=0) - cur_X_remaining[i+1:])
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=1)
        equal = diff_norm < th_num
        if np.mean(equal) == 0:
            unique_inds.append(i)

    unique_inds.append(len(cur_X_remaining)-1)

    if verbose:
        print('cur X filtering:',cur_X_remaining.shape[0], '->', len(unique_inds))

    if len(unique_inds) == 0:
        return []
    remaining_inds = remaining_inds[np.array(unique_inds)]


    return remaining_inds
