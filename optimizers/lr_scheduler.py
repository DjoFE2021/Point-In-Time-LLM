class LRScheduler:
    """
    Linear warmup → constant → linear warmdown learning-rate schedule.

    Parameters
    ----------
    num_iterations : int
        Total number of training iterations.
    warmdown_iters : int
        Number of final iterations over which the learning rate decays linearly to zero.
    warmup_iters : int
        Number of initial iterations used for linear warmup from 0 to 1.0 (scale factor).

    Attributes
    ----------
    num_iterations : int
        See *Parameters*.
    warmdown_iters : int
        See *Parameters*.
    warmup_iters : int
        See *Parameters*.
    """

    def __init__(self,
                 num_iterations : int, 
                 warmdown_iters : int,
                 warmup_iters   : int):
        """Initialize the scheduler (values stored as attributes)."""
        self.num_iterations = num_iterations
        self.warmdown_iters = warmdown_iters
        self.warmup_iters = warmup_iters
        
    def get_lr(self,
               it : int):
        """
        Return the LR scale factor for iteration ``it``.

        The schedule is:
        1. **Warmup**: linear ramp from 0 → 1 over ``warmup_iters``.
        2. **Plateau**: constant 1.0 until ``num_iterations - warmdown_iters``.
        3. **Warmdown**: linear decay from 1 → 0 over the last ``warmdown_iters`` steps.

        Parameters
        ----------
        it : int
            Zero-based training iteration index.

        Returns
        -------
        float
            Learning-rate multiplier in ``[0.0, 1.0]``.

        Raises
        ------
        AssertionError
            If ``it`` is greater than ``num_iterations``.
        """
        if it > self.num_iterations:
            return 0.01
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return (it+1) / self.warmup_iters
        # 2) constant lr for a while
        elif it < self.num_iterations - self.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (self.num_iterations - it) / self.warmdown_iters
            return decay_ratio
