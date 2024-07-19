from sklearn.ensemble import RandomForestClassifier
from TimeML.sklearn_extension.bootstrapping import *
class SequentialRandomForestClassifier(RandomForestClassifier):
    def _generate_sample_indices(self, random_state, n_samples, event):
        """Generate bootstrap sample indices with sequential bootstrap method."""
        random_instance = random_state  # get the RandomState instance

        ind_mat = get_indicator_matrix(
            event.index.to_series(),
            event['t1']
        )

        sample_indices = seq_bootstrap(ind_mat, n_samples)

        return sample_indices