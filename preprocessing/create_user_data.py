import os
import time
import psutil
import numpy as np
import pandas as pd

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

if __name__ == '__main__':
    start_time = time.time()
    print(f"[{time.ctime()}] Starting create_user_data.py (Memory: {get_memory_mb():.1f} MB)", flush=True)

    print("Loading raw CSV files...", flush=True)
    t0 = time.time()
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    order_products = pd.concat([prior_products, train_products], axis=0)
    products = pd.read_csv('../data/raw/products.csv')
    print(f"CSVs loaded in {time.time() - t0:.2f}s (Memory: {get_memory_mb():.1f} MB)", flush=True)

    print("Merging product taxonomy into order_products...", flush=True)
    t0 = time.time()
    order_products = order_products.merge(products[['product_id', 'aisle_id', 'department_id']], how='left', on='product_id')
    order_products = order_products.sort_values(['order_id', 'add_to_cart_order'])
    print(f"Merged and sorted order_products ({len(order_products):,} rows) in {time.time() - t0:.2f}s", flush=True)

    print("Single-pass fast grouping of order_products by order_id...", flush=True)
    t0 = time.time()
    
    order_ids = order_products['order_id'].values
    prods = order_products['product_id'].values.astype(str)
    reords = order_products['reordered'].values.astype(str)
    aisles = order_products['aisle_id'].values.astype(str)
    depts = order_products['department_id'].values.astype(str)

    op_order_ids = []
    op_prods = []
    op_reords = []
    op_aisles = []
    op_depts = []

    cur_order = -1
    cur_p = []
    cur_r = []
    cur_a = []
    cur_d = []

    for idx in range(len(order_ids)):
        o_id = order_ids[idx]
        if o_id != cur_order:
            if cur_order != -1:
                op_order_ids.append(cur_order)
                op_prods.append('_'.join(cur_p))
                op_reords.append('_'.join(cur_r))
                op_aisles.append('_'.join(cur_a))
                op_depts.append('_'.join(cur_d))
            cur_order = o_id
            cur_p = [prods[idx]]
            cur_r = [reords[idx]]
            cur_a = [aisles[idx]]
            cur_d = [depts[idx]]
        else:
            cur_p.append(prods[idx])
            cur_r.append(reords[idx])
            cur_a.append(aisles[idx])
            cur_d.append(depts[idx])

    if cur_order != -1:
        op_order_ids.append(cur_order)
        op_prods.append('_'.join(cur_p))
        op_reords.append('_'.join(cur_r))
        op_aisles.append('_'.join(cur_a))
        op_depts.append('_'.join(cur_d))

    op_df = pd.DataFrame({
        'order_id': op_order_ids,
        'product_id_str': op_prods,
        'reordered_str': op_reords,
        'aisle_id_str': op_aisles,
        'department_id_str': op_depts
    })
    print(f"op_grouped created ({len(op_df):,} orders) in {time.time() - t0:.2f}s (Memory: {get_memory_mb():.1f} MB)", flush=True)

    print("Merging order strings with orders.csv...", flush=True)
    t0 = time.time()
    orders = orders.sort_values(['user_id', 'order_number'])
    orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0).astype(int)
    
    df = orders.merge(op_df, how='left', on='order_id')
    for col in ['product_id_str', 'reordered_str', 'aisle_id_str', 'department_id_str']:
        df[col] = df[col].fillna('0').astype(str)
    print(f"Orders merged ({len(df):,} rows) in {time.time() - t0:.2f}s", flush=True)

    print("Single-pass fast grouping of orders by user_id...", flush=True)
    t0 = time.time()
    
    u_ids = df['user_id'].values
    o_ids = df['order_id'].values.astype(str)
    o_nums = df['order_number'].values.astype(str)
    o_dows = df['order_dow'].values.astype(str)
    o_hours = df['order_hour_of_day'].values.astype(str)
    o_days = df['days_since_prior_order'].values.astype(str)
    o_prods = df['product_id_str'].values
    o_aisles = df['aisle_id_str'].values
    o_depts = df['department_id_str'].values
    o_reords = df['reordered_str'].values
    o_evals = df['eval_set'].values

    out_users = []
    out_order_ids = []
    out_order_nums = []
    out_dows = []
    out_hours = []
    out_days = []
    out_prods = []
    out_aisles = []
    out_depts = []
    out_reords = []
    out_evals = []

    cur_user = -1
    c_oids = []
    c_onums = []
    c_dows = []
    c_hours = []
    c_days = []
    c_prods = []
    c_aisles = []
    c_depts = []
    c_reords = []
    c_eval = ''

    for idx in range(len(u_ids)):
        u_id = u_ids[idx]
        if u_id != cur_user:
            if cur_user != -1:
                out_users.append(cur_user)
                out_order_ids.append(' '.join(c_oids))
                out_order_nums.append(' '.join(c_onums))
                out_dows.append(' '.join(c_dows))
                out_hours.append(' '.join(c_hours))
                out_days.append(' '.join(c_days))
                out_prods.append(' '.join(c_prods))
                out_aisles.append(' '.join(c_aisles))
                out_depts.append(' '.join(c_depts))
                out_reords.append(' '.join(c_reords))
                out_evals.append(c_eval)
            cur_user = u_id
            c_oids = [o_ids[idx]]
            c_onums = [o_nums[idx]]
            c_dows = [o_dows[idx]]
            c_hours = [o_hours[idx]]
            c_days = [o_days[idx]]
            c_prods = [o_prods[idx]]
            c_aisles = [o_aisles[idx]]
            c_depts = [o_depts[idx]]
            c_reords = [o_reords[idx]]
            c_eval = o_evals[idx]
        else:
            c_oids.append(o_ids[idx])
            c_onums.append(o_nums[idx])
            c_dows.append(o_dows[idx])
            c_hours.append(o_hours[idx])
            c_days.append(o_days[idx])
            c_prods.append(o_prods[idx])
            c_aisles.append(o_aisles[idx])
            c_depts.append(o_depts[idx])
            c_reords.append(o_reords[idx])
            c_eval = o_evals[idx]

    if cur_user != -1:
        out_users.append(cur_user)
        out_order_ids.append(' '.join(c_oids))
        out_order_nums.append(' '.join(c_onums))
        out_dows.append(' '.join(c_dows))
        out_hours.append(' '.join(c_hours))
        out_days.append(' '.join(c_days))
        out_prods.append(' '.join(c_prods))
        out_aisles.append(' '.join(c_aisles))
        out_depts.append(' '.join(c_depts))
        out_reords.append(' '.join(c_reords))
        out_evals.append(c_eval)

    user_data = pd.DataFrame({
        'user_id': out_users,
        'order_ids': out_order_ids,
        'order_numbers': out_order_nums,
        'order_dows': out_dows,
        'order_hours': out_hours,
        'days_since_prior_orders': out_days,
        'product_ids': out_prods,
        'aisle_ids': out_aisles,
        'department_ids': out_depts,
        'reorders': out_reords,
        'eval_set': out_evals
    })
    print(f"User sequences created ({len(user_data):,} users) in {time.time() - t0:.2f}s (Memory: {get_memory_mb():.1f} MB)", flush=True)

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    output_path = '../data/processed/user_data.csv'
    t0 = time.time()
    user_data.to_csv(output_path, index=False)
    print(f"Finished writing {output_path} in {time.time() - t0:.2f}s. Total time: {time.time() - start_time:.2f}s", flush=True)
