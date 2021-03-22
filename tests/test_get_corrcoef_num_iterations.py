from gsa_framework.sensitivity_methods.correlations import (
    get_corrcoef_num_iterations,
)


def test_num_iterations():
    """
    test values are given in the following order:
    theta, interval_width, confidence_level=1-alpha, n_pearson, n_spearman, n_kendall
    where theta is the value of the correlation coefficient

    References
    ----------
    Sample size requirements for estimating Pearson, Kendall and Spearman correlations
    Bonett, Douglas G and Wright, Thomas A, 2000
    http://doi.org/10.1007/BF02294183
    Test values are taken from Table 1.
    """
    test = [
        (0.1, 0.1, 0.95, 1507, 1517, 661),
        (0.3, 0.1, 0.95, 1274, 1331, 560),
        (0.4, 0.2, 0.95, 273, 295, 122),
        (0.5, 0.3, 0.99, 168, 189, 76),
        (0.6, 0.2, 0.99, 276, 325, 123),
        (0.7, 0.3, 0.99, 82, 101, 39),
        (0.8, 0.1, 0.95, 205, 269, 93),
        (0.9, 0.2, 0.99, 34, 46, 18),
    ]

    for element in test:

        theta, width, conf_level, n_pearson, n_spearman, n_kendall = (
            element[0],
            element[1],
            element[2],
            element[3],
            element[4],
            element[5],
        )

        corrcoef_dict = get_corrcoef_num_iterations(
            theta=theta, interval_width=width, confidence_level=conf_level
        )

        n_pearson_comp = corrcoef_dict["pearson"]["num_iterations"]
        n_spearman_comp = corrcoef_dict["spearman"]["num_iterations"]
        n_kendall_comp = corrcoef_dict["kendall"]["num_iterations"]

        print(element)

        assert abs(n_pearson_comp - n_pearson) < 2
        assert abs(n_spearman_comp == n_spearman) < 2
        assert abs(n_kendall_comp == n_kendall) < 2
