import numpy as np


def random_drop(caption, script, image, drop_rate):
    """
    Randomly drop a value based on the drop rate.
    """
    # Randomly drop items but ensure at least one remains
    drop_caption = np.random.rand() < drop_rate
    drop_script = np.random.rand() < drop_rate
    drop_image = np.random.rand() < drop_rate

    # If all would be dropped, randomly keep one
    if drop_caption and drop_script and drop_image:
        keep_idx = np.random.randint(3)
        if keep_idx == 0:
            drop_caption = False
        elif keep_idx == 1:
            drop_script = False
        else:
            drop_image = False

    if drop_caption:
        caption = None
    if drop_script:
        script = None
    if drop_image:
        image = None

    return caption, script, image


class MLLMConditionedMixin:
    """
    Mixin for MLLM-conditioned models.

    Args:
        model_id: VLM.
        query_name: Name of the query type, e.g., "ss".
        num_queries: Number of queries to generate.
    """

    QUERY_TYPES = ["ss", "slat"]

    def __init__(
        self,
        *args,
        query_type: str = "ss",
        drop_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.query_type = query_type
        self.drop_rate = drop_rate

        if query_type not in self.QUERY_TYPES:
            raise ValueError(f"query_type must be 'ss' or 'slat', got: {query_type}")

        self.neg_outputs = self.training_models["mllm"](None, None, None, query_type)

    def random_drop(self, caption, script, image):
        """
        Randomly drop a value based on the drop rate.
        """
        return random_drop(caption, script, image, self.drop_rate)

    def get_cond(self, caption, script, image, **kwargs):
        """
        Get the script condition.
        """
        outputs = self.training_models["mllm"](caption, script, image, self.query_type)
        return outputs

    def get_inference_cond(self, caption, script, image):
        """
        Get the conditioning data for inference.
        """
        outputs = self.training_models["mllm"](caption, script, image, self.query_type)
        neg_outputs = self.neg_outputs.repeat(len(outputs), 1, 1)
        return {
            "cond": outputs,
            "neg_cond": neg_outputs,
        }
