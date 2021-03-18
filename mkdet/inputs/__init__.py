# from torch.utils.data import DataLoader

# import vin
# import augmentations

# # from .vin import *
# # from .augmentations import *


# def get_dataloader(cfgs, mode="train"):

#     if mode == "train":
#         transform = augmentations.train_multi_augment12
#         _dataset = VIN(cfgs, transform=transform, mode=mode)

#         get_dataset(cfgs, mode=mode)

#         _sampler = None
#         collate_fn = collater
#         _loader = InfiniteDataLoader(
#             dataset=_dataset,
#             batch_size=cfgs["batch_size"],
#             num_workers=cfgs["num_workers"],
#             pin_memory=True,
#             drop_last=False,
#             collate_fn=collate_fn,
#             sampler=_sampler,
#         )

#     else:
#         transform = None
#         _sampler = None
#         collate_fn = collater_test
#         _loader = InfiniteDataLoader(
#             dataset=_dataset,
#             batch_size=cfgs["batch_size"],
#             num_workers=cfgs["num_workers"],
#             pin_memory=True,
#             drop_last=False,
#             collate_fn=collate_fn,
#             sampler=_sampler,
#         )


# def get_dataset(cfgs, mode):
#     if mode == "train":
#         transform = get_augmentation(cfgs)
#     else:
#         transform = None

#     dataset = VIN(cfgs, transform=transform, mode=mode)

#     return dataset


# def get_augmentation(cfgs):
#     if cfgs["meta"]["inputs"]["augment"] == "train_multi_augment12":
#         aug_fn = train_multi_augment12
#     elif cfgs["meta"]["inputs"]["augment"]["augment"] is None:
#         aug_fn = None

#     return aug_fn


# def get_collater(mode="train"):
#     if mode == "train":
#         collate_fn = collater
#     else:
#         collate_fn = collater_test

#     return collate_fn
