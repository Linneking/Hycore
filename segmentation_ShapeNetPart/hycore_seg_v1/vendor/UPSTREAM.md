# DGCNN source provenance

Source: https://github.com/antao97/dgcnn.pytorch

Commit: `07d534c2702905010ec9991619f552d8cacae45b`

`dgcnn_model.py` is upstream `model.py`, with one compatibility change:
`device = torch.device('cuda')` is replaced by `device = x.device` in
`get_graph_feature`. Network mathematics, parameter initialization and shapes
are unchanged. Trailing whitespace is removed for repository hygiene. The
upstream MIT license is preserved in `LICENSE`.

This is a community DGCNN implementation listed by the original authors.
It is not an official HyCoRe-seg implementation.
