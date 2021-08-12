# Import Packages
import numpy as np
import pandas as pd
import random
np.random.seed(812)
random.seed(812)

# Create Supplier Dataframe Example
bau_order = ['Supplier C', 'Supplier A', 'Supplier B', 'Supplier D', 'Supplier F', 'Supplier E', 'Supplier G']
suppliers = [f'Supplier {x}' for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
bau_order = list(range(1, len(suppliers) + 1))
accept_prob = [random.choice(np.linspace(0,1,100)) for x in range(len(suppliers))]

supplier_df = pd.DataFrame({'Supplier' : suppliers, 'BAU_Order' : bau_order, 'Accept_Probability' : accept_prob})



# Define Function
def get_selection_probability(accept_probs):
    """
    Convert ordered individual probabilities into sequential probabilities
    E.g. Given ordered suppliers 1, 2, and 3 with acceptance probabilities 50%, 40%, and 30%,
    the odds of selecting suppliers 1, 2, and 3 are 0.5, 0.2, and 0.09 respectively
    Args:
        accept_probs (list): list of ordered acceptance probabilities
    """
    selection_prob = []
    for i, ap in enumerate(accept_probs):
        if i == 0:
            selection_prob.append(ap)
        else:
            selection_prob.append((1 - sum(selection_prob)) * ap)
    return selection_prob


# Execute Function
supplier_df['Selection_Probability'] = get_selection_probability(list(supplier_df['Accept_Probability']))













