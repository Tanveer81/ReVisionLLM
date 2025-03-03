"""
Utility: Convert textual cls feature and textual token feature into a single lmdb file for Ego4d-NLQ
"""

import numpy as np
import os
import tqdm
import lmdb
# import msgpack
# import msgpack_numpy
import io


# def dumps_msgpack(dump):
#     return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


if __name__ == '__main__':
    ###output path
    feature_save_path = "/home/atuin/v100dd/v100dd11/revisionllm/chapters_clipl14_features/lmdb"
    ###input load path
    cls_feature_load_path = "/home/atuin/v100dd/v100dd11/revisionllm/chapters_clipl14_features/chapters_clipl14_features"

    # token_feature_load_path = "..."
    text_output_env = lmdb.open(feature_save_path, map_size=1099511627776)

    print(len(os.listdir(cls_feature_load_path)))
    for item in tqdm.tqdm(os.listdir(cls_feature_load_path)):
        query_id = os.path.splitext(item)[0]
        # token_feature_path = os.path.join(token_feature_load_path, item)
        cls_feature_path = os.path.join(cls_feature_load_path, item)
        with text_output_env.begin(write=True) as text_output_txn:
            q_feat = np.load(cls_feature_path).astype(np.float32)
            # token_feat = np.load(token_feature_path).astype(np.float32)
            # features_dict = {"cls_features": q_feat[0], "token_features": token_feat}
            features_dict = {"features": q_feat}
            feature_dump = dumps_npz(features_dict, compress=True)
            text_output_txn.put(key=query_id.encode(), value=feature_dump)
