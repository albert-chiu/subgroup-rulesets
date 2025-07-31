import numpy as np
import pandas as pd
import operator
from bisect import bisect_left
import scipy.stats as stats
import matplotlib.pyplot as plt

def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    rules = []
    for child in idx:
        rule = []
        if child !=0:
            for node in recurse(left, right, child):
                rule.append(node)
            rules.append(rule)
    return rules

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
         return int(i-1)
    print('in find_lt,{}'.format(a))
    raise ValueError
    
def get_stats(df, Y, rules):
    covered = np.zeros(len(df))
    for rule in rules:
        val = np.array([not condition.endswith('_neg') for condition in rule]).astype(int)
        rule_cleaned = [condition.replace('_neg', '') for condition in rule]
        covered = covered + np.all(df[rule_cleaned]==val, axis=1)
    return np.mean(Y[covered > 0]), np.sum([x > 0 for x in covered])

def find_pareto_optimal_rule_sets(alpha_dict, X, tau, get_stats, tolerance=1e-10):
    """
    Find Pareto optimal rule sets from a dictionary of hyperparameter values to rule sets.
    Removes redundant rule sets with identical objective values.
    
    Args:
        alpha_dict: Dictionary where keys are alpha values and values are rule sets
        X: Parameter for get_stats function
        tau: Parameter for get_stats function  
        get_stats: Function that takes (X, tau, rule_set) and returns [effect, support]
        tolerance: Numerical tolerance for considering two values equal (default: 1e-10)
    
    Returns:
        Dictionary with alpha values as keys and corresponding Pareto optimal rule sets as values
    """
    # Step 1: Evaluate all rule sets and store results
    evaluated_rules = []
    for alpha, rule_set in alpha_dict.items():
        effect, support = get_stats(X, tau, rule_set)
        evaluated_rules.append({
            'alpha': alpha,
            'rule_set': rule_set,
            'effect': effect,
            'support': support
        })
    
    # Step 2: Remove redundant rule sets (same objective values)
    unique_rules = []
    seen_objectives = set()
    
    for rule in evaluated_rules:
        # Create a tuple of objectives rounded to handle floating point precision
        objective_key = (round(rule['effect'] / tolerance) * tolerance, 
                        round(rule['support'] / tolerance) * tolerance)
        
        if objective_key not in seen_objectives:
            seen_objectives.add(objective_key)
            unique_rules.append(rule)
        # If we've seen this objective combination, keep the one with smaller alpha
        else:
            # Find the existing rule with same objectives
            for i, existing in enumerate(unique_rules):
                existing_key = (round(existing['effect'] / tolerance) * tolerance,
                              round(existing['support'] / tolerance) * tolerance)
                if existing_key == objective_key:
                    if rule['alpha'] < existing['alpha']:
                        unique_rules[i] = rule
                    break
    
    # Step 3: Find Pareto optimal solutions among unique rules
    pareto_optimal = []
    
    for i, candidate in enumerate(unique_rules):
        is_dominated = False
        
        # Check if this candidate is dominated by any other solution
        for j, other in enumerate(unique_rules):
            if i != j:  # Don't compare with itself
                # Other dominates candidate if:
                # - Other is at least as good in both objectives AND
                # - Other is strictly better in at least one objective
                if (other['effect'] >= candidate['effect'] and 
                    other['support'] >= candidate['support'] and
                    (other['effect'] > candidate['effect'] or 
                     other['support'] > candidate['support'])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_optimal.append(candidate)
    
    # Step 4: Return as dictionary with alpha keys
    result = {}
    for item in pareto_optimal:
        result[item['alpha']] = item['rule_set']
    
    return result


def find_pareto_optimal_rule_sets(alpha_dict, X, tau, get_stats, tolerance=1e-10):
    """
    Find Pareto optimal rule sets from a dictionary of hyperparameter values to rule sets.
    Removes redundant rule sets with identical objective values.
    
    Args:
        alpha_dict: Dictionary where keys are alpha values and values are rule sets
        X: Parameter for get_stats function
        tau: Parameter for get_stats function  
        get_stats: Function that takes (X, tau, rule_set) and returns [effect, support]
        tolerance: Numerical tolerance for considering two values equal (default: 1e-10)
    
    Returns:
        Dictionary with alpha values as keys and corresponding Pareto optimal rule sets as values
    """
    # Step 1: Evaluate all rule sets and store results
    evaluated_rules = []
    for alpha, rule_set in alpha_dict.items():
        effect, support = get_stats(X, tau, rule_set)
        evaluated_rules.append({
            'alpha': alpha,
            'rule_set': rule_set,
            'effect': effect,
            'support': support
        })
    
    # Step 2: Remove redundant rule sets (same objective values)
    unique_rules = []
    seen_objectives = set()
    
    for rule in evaluated_rules:
        # Create a tuple of objectives rounded to handle floating point precision
        objective_key = (round(rule['effect'] / tolerance) * tolerance, 
                        round(rule['support'] / tolerance) * tolerance)
        
        if objective_key not in seen_objectives:
            seen_objectives.add(objective_key)
            unique_rules.append(rule)
        # If we've seen this objective combination, keep the one with smaller alpha
        else:
            # Find the existing rule with same objectives
            for i, existing in enumerate(unique_rules):
                existing_key = (round(existing['effect'] / tolerance) * tolerance,
                              round(existing['support'] / tolerance) * tolerance)
                if existing_key == objective_key:
                    if rule['alpha'] < existing['alpha']:
                        unique_rules[i] = rule
                    break
    
    # Step 3: Find Pareto optimal solutions among unique rules
    pareto_optimal = []
    
    for i, candidate in enumerate(unique_rules):
        is_dominated = False
        
        # Check if this candidate is dominated by any other solution
        for j, other in enumerate(unique_rules):
            if i != j:  # Don't compare with itself
                # Other dominates candidate if:
                # - Other is at least as good in both objectives AND
                # - Other is strictly better in at least one objective
                if (other['effect'] >= candidate['effect'] and 
                    other['support'] >= candidate['support'] and
                    (other['effect'] > candidate['effect'] or 
                     other['support'] > candidate['support'])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_optimal.append(candidate)
    
    # Step 4: Return as dictionary with alpha keys
    result = {}
    for item in pareto_optimal:
        result[item['alpha']] = item['rule_set']
    
    return result

def create_boxplots(X, W, rs, Y=None, tau=None, power_df=None, min_distance_y=20.0, 
                         box_width=0.04, max_support=None, min_support=None,
                         min_gap=0.10, scale="outcome", quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                         support_figures=2, line=False, show_power="labels"):
    """
    Clean boxplot function with two modes:
    1. tau: vector of treatment effects (plot directly)
    2. Y: raw outcomes (calculate difference in means)
    
    Parameters:
    -----------
    X : DataFrame
        Features
    W : array-like
        Treatment indicators (0/1)
    rs : dict
        Dictionary of rulesets
    Y : array-like, optional
        Raw outcomes (for difference-in-means calculation)
    tau : array-like, optional
        Individual treatment effects (plot directly)
    """
    # Check that exactly one of Y or tau is provided
    if Y is None and tau is None:
        raise ValueError("Either Y (raw outcomes) or tau (treatment effects) must be provided")
    if Y is not None and tau is not None:
        raise ValueError("Provide either Y or tau, not both")
    
    # Determine mode
    if tau is not None:
        mode = "tau"
        data_vector = np.array(tau)
    else:
        mode = "Y"
        data_vector = np.array(Y)
    
    # Calculate data for all rulesets
    all_rulesets = []
    
    # Calculate standard deviation for scaling if needed
    if scale == "sd":
        if mode == "tau":
            outcome_std = np.std(tau)
        else:
            outcome_std = np.std(Y)
    
    for rs_name, ruleset in rs.items():
        try:
            mask = evaluate_ruleset(X, ruleset)
            
            if np.sum(mask) == 0:
                continue
                
            data_subset = data_vector[mask]
            W_subset = W[mask]
            
            data_subset = np.array(data_subset)
            W_subset = np.array(W_subset)
            
            if mode == "tau":
                # Mode 1: tau provided - plot treatment effects directly
                # For tau, we typically want treated units only, or all units
                # Let's use all units in the subgroup
                treatment_effects = data_subset
                avg_effect = np.mean(data_subset)
                
            else:
                # Mode 2: Y provided - calculate difference in means
                Y_treated = data_subset[W_subset == 1]
                Y_control = data_subset[W_subset == 0]
                
                if len(Y_treated) == 0 or len(Y_control) == 0:
                    continue
                    
                avg_effect = np.mean(Y_treated) - np.mean(Y_control)
                control_mean = np.mean(Y_control)
                
                if len(Y_treated) == 1:
                    treatment_effects = np.array([Y_treated[0] - control_mean])
                else:
                    treatment_effects = Y_treated - control_mean
            
            support_size = np.sum(mask) / len(X)
            
            if max_support is not None and support_size > max_support:
                continue
            if min_support is not None and support_size < min_support:
                continue
            
            if len(treatment_effects) == 0:
                continue
            
            # Scale treatment effects if requested
            if scale == "sd":
                treatment_effects = treatment_effects / outcome_std
                avg_effect = avg_effect / outcome_std
            
            power_val = 0
            if power_df is not None:
                power_row = power_df[power_df['ruleset'] == rs_name]
                power_val = power_row['power'].iloc[0] if len(power_row) > 0 else 0
            
            all_rulesets.append({
                'name': rs_name,
                'avg_effect': avg_effect,
                'support_size': support_size,
                'treatment_effects': treatment_effects,
                'power': power_val
            })
            
        except Exception as e:
            continue
    
    if len(all_rulesets) == 0:
        return [], [], [], [], [], None, None, None, None, None
    
    # Sort by effect size (highest first)
    all_rulesets.sort(key=lambda x: x['avg_effect'], reverse=True)
    
    # Apply spacing logic
    selected_rulesets = []
    
    for ruleset in all_rulesets:
        x_pos = ruleset['support_size']
        y_pos = ruleset['avg_effect']
        
        # Calculate box boundaries
        box_start = x_pos - box_width/2
        box_end = x_pos + box_width/2
        
        # Check conflicts
        conflicts = False
        for selected in selected_rulesets:
            prev_x = selected['support_size']
            prev_y = selected['avg_effect']
            prev_start = prev_x - box_width/2
            prev_end = prev_x + box_width/2
            
            # X-overlap check
            x_overlaps = not (box_end + min_gap < prev_start or box_start > prev_end + min_gap)
            # Y-distance check  
            y_too_close = abs(y_pos - prev_y) < min_distance_y
            
            # Conflict if EITHER x overlaps AND y too close
            if x_overlaps or y_too_close:
                conflicts = True
                break
        
        if not conflicts:
            selected_rulesets.append(ruleset)
    
    # Sort by support for plotting
    selected_rulesets.sort(key=lambda x: x['support_size'])
    
    final_data = [rs['treatment_effects'] for rs in selected_rulesets]
    final_positions = [rs['support_size'] for rs in selected_rulesets]
    final_powers = [rs['power'] for rs in selected_rulesets]
    final_names = [rs['name'] for rs in selected_rulesets]
    final_effects = [rs['avg_effect'] for rs in selected_rulesets]
    
    alphas = [x['name'] for x in selected_rulesets]
    
    # Return all the plotting parameters
    return alphas, final_data, final_positions, final_powers, final_names, final_effects, quantiles, scale, support_figures, line, show_power

def calculate_ruleset_power(X_train, W_train, X_test, rs_dict, estimated_effects, alpha=0.05, sigma=None, Y_train=None):
    """
    Calculate statistical power for each ruleset 
    """
    def power_calculation_two_sample(effect_size, n1, n2, sigma1=1, sigma2=1, alpha=0.05, alternative='two-sided'):
        """
        Calculate statistical power for two-sample t-test - robust version
        """
        # Input validation
        if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in [effect_size, n1, n2, sigma1, sigma2, alpha]):
            print(f"Invalid inputs: effect_size={effect_size}, n1={n1}, n2={n2}, sigma1={sigma1}, sigma2={sigma2}, alpha={alpha}")
            return np.nan
        
        if n1 <= 1 or n2 <= 1 or sigma1 <= 0 or sigma2 <= 0:
            return np.nan
        
        try:
            # Convert to float to ensure numeric types
            effect_size = float(effect_size)
            n1, n2 = int(n1), int(n2)
            sigma1, sigma2 = float(sigma1), float(sigma2)
            alpha = float(alpha)
            
            # Pooled standard error
            pooled_se = np.sqrt(sigma1**2/n1 + sigma2**2/n2)
            
            if pooled_se == 0:
                return np.nan
            
            # Degrees of freedom
            df = n1 + n2 - 2
            
            # Critical value
            if alternative == 'two-sided':
                critical_t = stats.t.ppf(1 - alpha/2, df=df)
            elif alternative == 'greater':
                critical_t = stats.t.ppf(1 - alpha, df=df)
            else:  # 'less'
                critical_t = stats.t.ppf(alpha, df=df)
            
            # Non-centrality parameter
            ncp = effect_size / pooled_se
            
            # Power calculation
            if alternative == 'two-sided':
                power = 1 - stats.t.cdf(critical_t, df=df, loc=ncp) + stats.t.cdf(-critical_t, df=df, loc=ncp)
            elif alternative == 'greater':
                power = 1 - stats.t.cdf(critical_t, df=df, loc=ncp)
            else:  # 'less'
                power = stats.t.cdf(critical_t, df=df, loc=ncp)
            
            return float(power)
            
        except Exception as e:
            print(f"Error in power calculation: {e}")
            return np.nan

    if sigma is None:
        all_treated_outcomes = []
        all_control_outcomes = []

        for ruleset in rs_dict.values():
            mask = _evaluate_ruleset(X_train, ruleset)
            
            if np.sum(mask) == 0:
                continue
                
            Y_subset = Y_train[mask]
            W_subset = W_train[mask]
            
            treated_outcomes = Y_subset[W_subset == 1]
            control_outcomes = Y_subset[W_subset == 0]
            
            if len(treated_outcomes) > 1:
                all_treated_outcomes.extend(treated_outcomes)
            if len(control_outcomes) > 1:
                all_control_outcomes.extend(control_outcomes)
        
        sigma1 = np.std(all_treated_outcomes) if len(all_treated_outcomes) > 1 else 1
        sigma2 = np.std(all_control_outcomes) if len(all_control_outcomes) > 1 else 1
        
    else: 
        sigma1 = sigma[0]
        sigma2 = sigma[1]

    n_train = len(X_train)
    n_test = len(X_test)
    
    results = []
    
    for i, (rs_name, ruleset) in enumerate(rs_dict.items()):
        try:
            # Find observations covered by this ruleset in training set
            mask_train = _evaluate_ruleset(X_train, ruleset)
            
            if np.sum(mask_train) == 0:
                results.append({
                    'ruleset': rs_name,
                    'n1_train': 0,
                    'n0_train': 0,
                    'n1_test_projected': 0,
                    'n0_test_projected': 0,
                    'estimated_effect': np.nan,
                    'power': np.nan,
                    'coverage_rate_train': 0
                })
                continue
            
            # Count treated and control in training set for this ruleset
            W_covered = W_train[mask_train]
            n1_train = int(np.sum(W_covered == 1))  # Treated covered in training
            n0_train = int(np.sum(W_covered == 0))  # Control covered in training
            
            # Calculate coverage rates
            coverage_rate = float(np.sum(mask_train) / n_train)
            treatment_rate_in_covered = float(n1_train / np.sum(mask_train)) if np.sum(mask_train) > 0 else 0
            
            # Project to test set
            n_covered_test = coverage_rate * n_test
            n1_test_projected = int(n_covered_test * treatment_rate_in_covered)
            n0_test_projected = int(n_covered_test * (1 - treatment_rate_in_covered))
            
            # Get estimated effect for this ruleset
            if isinstance(estimated_effects, dict):
                effect_size = estimated_effects.get(rs_name, np.nan)
            else:
                # Assume estimated_effects is array-like
                effect_size = estimated_effects[i] if i < len(estimated_effects) else np.nan
            
            # Ensure effect_size is a single number
            if isinstance(effect_size, (list, tuple, np.ndarray)):
                effect_size = effect_size[0] if len(effect_size) > 0 else np.nan
            
            # Calculate power if we have valid inputs
            if (not np.isnan(effect_size) and n1_test_projected > 1 and n0_test_projected > 1):
                power = power_calculation_two_sample(
                    effect_size=effect_size,
                    n1=n1_test_projected,
                    n2=n0_test_projected,
                    sigma1=sigma1,
                    sigma2=sigma2,
                    alpha=alpha,
                    alternative='two-sided'
                )
            else:
                power = np.nan
            
            results.append({
                'ruleset': rs_name,
                'n1_train': n1_train,
                'n0_train': n0_train,
                'n1_test_projected': n1_test_projected,
                'n0_test_projected': n0_test_projected,
                'coverage_rate_train': coverage_rate,
                'treatment_rate_in_covered': treatment_rate_in_covered,
                'estimated_effect': float(effect_size) if not np.isnan(effect_size) else np.nan,
                'power': power
            })
            
        except Exception as e:
            print(f"Error processing ruleset {rs_name}: {e}")
            results.append({
                'ruleset': rs_name,
                'n1_train': 0,
                'n0_train': 0,
                'n1_test_projected': 0,
                'n0_test_projected': 0,
                'estimated_effect': np.nan,
                'power': np.nan,
                'coverage_rate_train': 0
            })
    
    return pd.DataFrame(results)


def _evaluate_ruleset(X, ruleset):
            """
            Evaluate which observations satisfy the ruleset
            
            Ruleset format: list of lists, where:
            - Each inner list is a conjunction (AND) of features
            - The outer list is a disjunction (OR) of these conjunctions
            - Features ending in '_neg' are negated
            
            Example: [['CURRJOB_1', 'AGE_L17', 'RACE_BLACK_neg']]
            means: CURRJOB_1 AND AGE_L17 AND (NOT RACE_BLACK)
            """
            if not ruleset or len(ruleset) == 0:
                return np.zeros(len(X), dtype=bool)
            
            # Start with all False - we'll OR the conditions
            final_mask = np.zeros(len(X), dtype=bool)
            
            # Each inner list represents an AND condition
            for conjunction in ruleset:
                # Start with all True for this conjunction
                conjunction_mask = np.ones(len(X), dtype=bool)
                
                # Apply each feature in the conjunction
                for feature in conjunction:
                    if feature.endswith('_neg'):
                        # Negated feature - remove '_neg' suffix and negate
                        base_feature = feature[:-4]
                        if base_feature in X.columns:
                            conjunction_mask &= (X[base_feature] == 0)  # Assuming binary features
                        else:
                            # Feature not found - this conjunction fails
                            conjunction_mask = np.zeros(len(X), dtype=bool)
                            break
                    else:
                        # Regular feature
                        if feature in X.columns:
                            conjunction_mask &= (X[feature] == 1)  # Assuming binary features
                        else:
                            # Feature not found - this conjunction fails
                            conjunction_mask = np.zeros(len(X), dtype=bool)
                            break
                
                # OR this conjunction with the final result
                final_mask |= conjunction_mask
            
            return final_mask

def create_boxplots(X, W, rs, Y=None, tau=None, power_df=None, min_distance_y=20.0, 
                         box_width=0.04, max_support=None, min_support=None,
                         min_gap=0.10, scale="outcome", quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                         support_figures=2, line=False, show_power="labels"):
    """
    Clean boxplot function with two modes:
    1. tau: vector of treatment effects (plot directly)
    2. Y: raw outcomes (calculate difference in means)
    
    Parameters:
    -----------
    X : DataFrame
        Features
    W : array-like
        Treatment indicators (0/1)
    rs : dict
        Dictionary of rulesets
    Y : array-like, optional
        Raw outcomes (for difference-in-means calculation)
    tau : array-like, optional
        Individual treatment effects (plot directly)
    """
    # Check that exactly one of Y or tau is provided
    if Y is None and tau is None:
        raise ValueError("Either Y (raw outcomes) or tau (treatment effects) must be provided")
    if Y is not None and tau is not None:
        raise ValueError("Provide either Y or tau, not both")
    
    # Determine mode
    if tau is not None:
        mode = "tau"
        data_vector = np.array(tau)
    else:
        mode = "Y"
        data_vector = np.array(Y)
    
    # Calculate data for all rulesets
    all_rulesets = []
    
    # Calculate standard deviation for scaling if needed
    if scale == "sd":
        if mode == "tau":
            outcome_std = np.std(tau)
        else:
            outcome_std = np.std(Y)
    
    for rs_name, ruleset in rs.items():
        try:
            mask = _evaluate_ruleset(X, ruleset)
            
            if np.sum(mask) == 0:
                continue
                
            data_subset = data_vector[mask]
            W_subset = W[mask]
            
            data_subset = np.array(data_subset)
            W_subset = np.array(W_subset)
            
            if mode == "tau":
                # Mode 1: tau provided - plot treatment effects directly
                # For tau, we typically want treated units only, or all units
                # Let's use all units in the subgroup
                treatment_effects = data_subset
                avg_effect = np.mean(data_subset)
                
            else:
                # Mode 2: Y provided - calculate difference in means
                Y_treated = data_subset[W_subset == 1]
                Y_control = data_subset[W_subset == 0]
                
                if len(Y_treated) == 0 or len(Y_control) == 0:
                    continue
                    
                avg_effect = np.mean(Y_treated) - np.mean(Y_control)
                control_mean = np.mean(Y_control)
                
                if len(Y_treated) == 1:
                    treatment_effects = np.array([Y_treated[0] - control_mean])
                else:
                    treatment_effects = Y_treated - control_mean
            
            support_size = np.sum(mask) / len(X)
            
            if max_support is not None and support_size > max_support:
                continue
            if min_support is not None and support_size < min_support:
                continue
            
            if len(treatment_effects) == 0:
                continue
            
            # Scale treatment effects if requested
            if scale == "sd":
                treatment_effects = treatment_effects / outcome_std
                avg_effect = avg_effect / outcome_std
            
            power_val = 0
            if power_df is not None:
                power_row = power_df[power_df['ruleset'] == rs_name]
                power_val = power_row['power'].iloc[0] if len(power_row) > 0 else 0
            
            all_rulesets.append({
                'name': rs_name,
                'avg_effect': avg_effect,
                'support_size': support_size,
                'treatment_effects': treatment_effects,
                'power': power_val
            })
            
        except Exception as e:
            continue
    
    if len(all_rulesets) == 0:
        return [], [], [], [], [], None, None, None, None, None, None
    
    # Sort by effect size (highest first)
    all_rulesets.sort(key=lambda x: x['avg_effect'], reverse=True)
    
    # Apply spacing logic
    selected_rulesets = []
    
    for ruleset in all_rulesets:
        x_pos = ruleset['support_size']
        y_pos = ruleset['avg_effect']
        
        # Calculate box boundaries
        box_start = x_pos - box_width/2
        box_end = x_pos + box_width/2
        
        # Check conflicts
        conflicts = False
        for selected in selected_rulesets:
            prev_x = selected['support_size']
            prev_y = selected['avg_effect']
            prev_start = prev_x - box_width/2
            prev_end = prev_x + box_width/2
            
            # X-overlap check
            x_overlaps = not (box_end + min_gap < prev_start or box_start > prev_end + min_gap)
            # Y-distance check  
            y_too_close = abs(y_pos - prev_y) < min_distance_y
            
            # Conflict if EITHER x overlaps AND y too close
            if x_overlaps or y_too_close:
                conflicts = True
                break
        
        if not conflicts:
            selected_rulesets.append(ruleset)
    
    # Sort by support for plotting
    selected_rulesets.sort(key=lambda x: x['support_size'])
    
    final_data = [rs['treatment_effects'] for rs in selected_rulesets]
    final_positions = [rs['support_size'] for rs in selected_rulesets]
    final_powers = [rs['power'] for rs in selected_rulesets]
    final_names = [rs['name'] for rs in selected_rulesets]
    final_effects = [rs['avg_effect'] for rs in selected_rulesets]
    
    alphas = [x['name'] for x in selected_rulesets]
    
    # Return all the plotting parameters
    return alphas, final_data, final_positions, final_powers, final_names, final_effects, quantiles, scale, support_figures, line, show_power

def _evaluate_ruleset(X, ruleset):
            """
            Evaluate which observations satisfy the ruleset
            
            Ruleset format: list of lists, where:
            - Each inner list is a conjunction (AND) of features
            - The outer list is a disjunction (OR) of these conjunctions
            - Features ending in '_neg' are negated
            
            Example: [['CURRJOB_1', 'AGE_L17', 'RACE_BLACK_neg']]
            means: CURRJOB_1 AND AGE_L17 AND (NOT RACE_BLACK)
            """
            if not ruleset or len(ruleset) == 0:
                return np.zeros(len(X), dtype=bool)
            
            # Start with all False - we'll OR the conditions
            final_mask = np.zeros(len(X), dtype=bool)
            
            # Each inner list represents an AND condition
            for conjunction in ruleset:
                # Start with all True for this conjunction
                conjunction_mask = np.ones(len(X), dtype=bool)
                
                # Apply each feature in the conjunction
                for feature in conjunction:
                    if feature.endswith('_neg'):
                        # Negated feature - remove '_neg' suffix and negate
                        base_feature = feature[:-4]
                        if base_feature in X.columns:
                            conjunction_mask &= (X[base_feature] == 0)  # Assuming binary features
                        else:
                            # Feature not found - this conjunction fails
                            conjunction_mask = np.zeros(len(X), dtype=bool)
                            break
                    else:
                        # Regular feature
                        if feature in X.columns:
                            conjunction_mask &= (X[feature] == 1)  # Assuming binary features
                        else:
                            # Feature not found - this conjunction fails
                            conjunction_mask = np.zeros(len(X), dtype=bool)
                            break
                
                # OR this conjunction with the final result
                final_mask |= conjunction_mask
            
            return final_mask