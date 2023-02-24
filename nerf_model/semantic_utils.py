import torch

class SemanticRemap:
    def __init__(self):
        self.direct_remap_dict = None
        self.inv_remap_dict = None

    def __apply_remap(self, semantic_images, inv=False):
        remap_to_apply = self.direct_remap_dict if not inv else self.inv_remap_dict
        for c, c_remap in remap_to_apply.items():
            semantic_images[semantic_images == c] = c_remap
        return semantic_images

    def __create_direct_inv_remap(self, semantic_images):
        print("[INFO] Start semantic remap ->\n\t", end="")

        # get semantic classes
        self.semantic_classes = torch.unique(semantic_images)

        # class <-> index
        self.direct_remap_dict = dict(zip(self.semantic_classes, torch.arange(len(self.semantic_classes))))

        # index <-> class
        self.inv_remap_dict = {item: key for key, item in self.direct_remap_dict.items()}

        print(*[f"{k}: {v}" for (k, v) in self.direct_remap_dict.items()], sep='\n\t')

    def remap(self, semantic_images, inplace=False):
        if self.direct_remap_dict is None:
            self.__create_direct_inv_remap(semantic_images)
        if not inplace:
            semantic_images = semantic_images.copy()
        return self.__apply_remap(semantic_images, inv=False)

    def inv_remap(self, semantic_images, inplace=False):
        if not inplace:
            semantic_images = semantic_images.copy()
        return self.__apply_remap(semantic_images, inv=True)
