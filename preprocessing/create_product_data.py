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
    print(f"[{time.ctime()}] Starting create_product_data.py (Memory: {get_memory_mb():.1f} MB)", flush=True)

    print("Loading user_data.csv...", flush=True)
    t0 = time.time()
    df = pd.read_csv('../data/processed/user_data.csv')
    print(f"user_data.csv loaded ({len(df):,} users) in {time.time() - t0:.2f}s (Memory: {get_memory_mb():.1f} MB)", flush=True)

    out_user_id = []
    out_product_id = []
    out_is_ordered = []
    out_index_in_order = []
    out_order_size = []
    out_reorder_size = []
    out_order_num = []
    out_dow = []
    out_hour = []
    out_days = []
    out_label = []
    out_eval = []

    print("Processing 206,209 users with fast O(1) history sets...", flush=True)
    t0 = time.time()

    for row in df.itertuples(index=False):
        u_id = row.user_id
        eval_set = str(row.eval_set)
        
        prods_split = str(row.product_ids).split()
        reords_split = str(row.reorders).split()
        
        prods_hist = prods_split[:-1]
        next_prods_str = prods_split[-1] if len(prods_split) > 0 else ""
        next_prods_set = set(next_prods_str.split('_')) if next_prods_str else set()

        reords_hist = reords_split[:-1]

        order_sets = []
        order_maps = []
        order_sizes_list = []
        for o_str in prods_hist:
            items = o_str.split('_')
            order_sizes_list.append(str(len(items)))
            o_set = set(items)
            o_map = {p: str(idx + 1) for idx, p in enumerate(items)}
            order_sets.append(o_set)
            order_maps.append(o_map)

        reorder_sizes_list = []
        for r_str in reords_hist:
            r_items = r_str.split('_')
            reorder_sizes_list.append(str(sum(1 for x in r_items if x == '1')))

        order_size_str = ' '.join(order_sizes_list)
        reorder_size_str = ' '.join(reorder_sizes_list)

        # Slice temporal context to strictly history (N-1)
        onum_str = ' '.join(str(row.order_numbers).split()[:-1])
        dow_str = ' '.join(str(row.order_dows).split()[:-1])
        hour_str = ' '.join(str(row.order_hours).split()[:-1])
        days_str = ' '.join(str(row.days_since_prior_orders).split()[:-1])

        product_set = set()
        for o_set in order_sets:
            product_set.update(o_set)

        for p_id_str in product_set:
            is_ord = ' '.join('1' if p_id_str in o_set else '0' for o_set in order_sets)
            idx_in_ord = ' '.join(o_map.get(p_id_str, '0') for o_map in order_maps)
            lbl = 1 if p_id_str in next_prods_set else 0

            out_user_id.append(u_id)
            out_product_id.append(int(p_id_str))
            out_is_ordered.append(is_ord)
            out_index_in_order.append(idx_in_ord)
            out_order_size.append(order_size_str)
            out_reorder_size.append(reorder_size_str)
            out_order_num.append(onum_str)
            out_dow.append(dow_str)
            out_hour.append(hour_str)
            out_days.append(days_str)
            out_label.append(lbl)
            out_eval.append(eval_set)

        # Add None product (0)
        zeros_str = ' '.join(['0'] * len(order_sets))
        none_lbl = 1 if len(next_prods_set) == 0 else 0
        out_user_id.append(u_id)
        out_product_id.append(0)
        out_is_ordered.append(zeros_str)
        out_index_in_order.append(zeros_str)
        out_order_size.append(order_size_str)
        out_reorder_size.append(reorder_size_str)
        out_order_num.append(onum_str)
        out_dow.append(dow_str)
        out_hour.append(hour_str)
        out_days.append(days_str)
        out_label.append(none_lbl)
        out_eval.append(eval_set)

    print(f"Generated {len(out_user_id):,} candidate rows in {time.time() - t0:.2f}s", flush=True)

    print("Creating DataFrame and saving to CSV...", flush=True)
    t0 = time.time()
    out_df = pd.DataFrame({
        'user_id': out_user_id,
        'product_id': out_product_id,
        'is_ordered_history': out_is_ordered,
        'index_in_order_history': out_index_in_order,
        'order_size_history': out_order_size,
        'reorder_size_history': out_reorder_size,
        'order_dow_history': out_dow,
        'order_hour_history': out_hour,
        'days_since_prior_order_history': out_days,
        'order_number_history': out_order_num,
        'label': out_label,
        'eval_set': out_eval
    })

    print("Merging with products.csv taxonomy...", flush=True)
    products = pd.read_csv('../data/raw/products.csv')
    none_row = pd.DataFrame({'product_id': [0], 'product_name': ['none'], 'aisle_id': [0], 'department_id': [0]})
    products = pd.concat([products, none_row], ignore_index=True)
    out_df = out_df.merge(products[['product_id', 'aisle_id', 'department_id', 'product_name']], on='product_id', how='left')
    out_df['product_name'] = out_df['product_name'].fillna('none')
    out_df['aisle_id'] = out_df['aisle_id'].fillna(0).astype(int)
    out_df['department_id'] = out_df['department_id'].fillna(0).astype(int)

    # Reorder columns to match original schema
    cols_order = [
        'user_id', 'product_id', 'aisle_id', 'department_id', 'product_name',
        'is_ordered_history', 'index_in_order_history', 'order_size_history',
        'reorder_size_history', 'order_dow_history', 'order_hour_history',
        'days_since_prior_order_history', 'order_number_history', 'label', 'eval_set'
    ]
    out_df = out_df[cols_order]

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')
        
    output_path = '../data/processed/product_data.csv'
    out_df.to_csv(output_path, index=False)
    print(f"Finished writing {output_path} in {time.time() - t0:.2f}s. Total time: {time.time() - start_time:.2f}s", flush=True)
