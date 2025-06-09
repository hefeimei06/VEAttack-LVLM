import torch


def attention_map(embedding, image_features, text_features):

    features = torch.cat((embedding.unsqueeze(1), image_features), dim=1)
    features = features / features.norm(dim=1, keepdim=True)
    img_spatial_feat = features[:, 1:, :]

    """Text guided attention map"""
    am = img_spatial_feat @ text_features.unsqueeze(-1)
    am = (am - am.min(1, keepdim=True)[0]) / (am.max(1, keepdim=True)[0] - am.min(1, keepdim=True)[0])
    """reshape"""
    side = int(am.shape[1] ** 0.5)
    am = am.reshape(am.shape[0], side, side, -1).permute(0, 3, 1, 2)

    """interpolate"""
    # am = torch.nn.functional.interpolate(am, 224, mode='bilinear')
    return am
