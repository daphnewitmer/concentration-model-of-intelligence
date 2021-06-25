import matplotlib.pyplot as plt
import numpy as np
from src import parameters as params
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

class Test:
    def __init__(self, simulation):
        self.simulation = simulation

    def run(self):

        # self.plot_cog_cap_structure()

        # Make plots of general learning results and the test structure
        fig, axs = plt.subplots(2, 3)
        self.plot_knowledge_structure(axs, (0, 0), 1)
        self.plot_item_variation('knowledge', axs, (0, 1), 2)  # knowledge or test
        self.plot_item_variation('test', axs, (0, 2), 3)  # knowledge or test
        self.plot_skills_learned(axs, (1, 0), 4)
        self.plot_learning_correlations(axs, (1, 1), 5)
        self.plot_test_structure(axs, (1, 2), 6)
        parameters = 'Parameters: \nPercentage random: ' + str(params.PERC_RAND) \
                     + '\nAcquired knowledge weight: ' + str(params.ACQ_KNOWL_WEIGHT) \
                     + '%\nStart percentage cog cap: ' + str(params.START_PERC_COG_CAP) + '%'
        plt.figtext(0.01, 0.9, parameters)
        plt.figtext(0.5, 0.96, 'Learning results and iq test structure', weight='bold')
        plt.show()

        # Make separate overview with plots of test results
        fig, axs = plt.subplots(4, len(params.TEST_AGES))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        plt.figtext(0.5, 0.96, 'IQ-test results', weight='bold')

        for i in range(len(params.TEST_AGES)):
            plt.figtext(0.2 * (i+1), 0.93, 'Test ' + str(i+1), weight='bold')
            start = 1000 + (i * 100)
            self.plot_corr_test_items(axs, (0, i), i+1, i + 1)
            self.plot_raw_iq_scores(axs, (1, i), ". Frequency Hist", start, start + 100, 4 + i + 1)
            self.plot_corr_iq_scores(axs, (2, i), 'cog_cap', i+1, 8 + i + 1)
            self.plot_corr_iq_scores(axs, (3, i), 'concentration', i+1, 12 + i + 1)
            # self.factor_analysis(i+1)

        plt.show()

    def plot_knowledge_structure(self, axs, place, plot_nr):
        """
        Plot with required cognitive capacity and average cognitive capacity for m in knowledge matrix
        """

        person = 0

        cog_cap = self.simulation.get_cog_cap(person)
        conc = self.simulation.concentration_matrix[:, person]
        axs[place].plot(np.arange(params.nrOfMicroskillsNormal), cog_cap,
                        label='Person (p=' + str(person) + ') Cognitive Capacity')
        axs[place].plot(np.arange(params.nrOfMicroskillsNormal), conc,
                        label='Person (p=' + str(person) + ') Concentration', zorder=1, color='bisque')

        axs[place].plot(np.arange(params.nrOfMicroskillsNormal),
                        self.simulation.knowledge_matrix[:, 0],  # [self.simulation.schooling_matrix[person,:]]
                        label='Required Cognitive Capacity',
                        zorder=0,
                        color='paleturquoise')
        axs[place].set_xlabel('Microskills')
        axs[place].set_ylabel('Cognitive Capacity')
        axs[place].set_title(str(plot_nr) + ': Knowledge Structure')
        axs[place].legend()

    def plot_item_variation(self, matrix, axs, place, plot_nr):
        """ Plot sum knowledge to check variation between skills """

        if matrix == 'test':
            matrix = self.simulation.test_matrix
            title = str(plot_nr) + ': Variation Between Item Connectedness (IQ-test)'
        elif matrix == 'knowledge':
            matrix = self.simulation.knowledge_matrix
            title = str(plot_nr) + ': Variation Between Item Connectedness (General)'

        matrix_without_cog_cap = matrix[:, 1:]
        similarity_matrix = matrix_without_cog_cap.dot(matrix_without_cog_cap.T)
        sum_knowl_array = similarity_matrix.sum(axis=0)

        axs[place].plot(sum_knowl_array)
        axs[place].set_xlabel('Microskills')
        axs[place].set_ylabel('Sum Knowledge')
        axs[place].set_title(title)

    def plot_skills_learned(self, axs, place, plot_nr):
        """" Plot total number of skills learned per timepoint """

        start_p = int(0)
        last_p = int(10)
        nr_of_test_items = len(params.TEST_AGES * params.nrOfMicroskillsIQ)

        learning_matrix_without_test_select_pers = self.simulation.learning_matrix[:,
                                                   start_p:last_p]  # microskill, person
        learned_skills_per_timepoint_matrix = np.cumsum(learning_matrix_without_test_select_pers, axis=0)

        for person in range(last_p):
            axs[place].plot(learned_skills_per_timepoint_matrix[:-nr_of_test_items, person], label='Person: ' + str(person))

        axs[place].set_title(str(plot_nr) + ": Microskills Learned per Timepoint")
        axs[place].set_xlabel('Timepoints')
        axs[place].set_ylabel('Nr of Learned Microskills')
        axs[place].legend()

    def plot_learning_correlations(self, axs, place, plot_nr):
        """ Correlations with sum of learned skills and cognitive capacity and concentration """

        learning_matrix_without_test = self.simulation.learning_matrix[:-100, :]  # microskill, person
        learned_skills_per_timepoint_matrix = np.cumsum(learning_matrix_without_test, axis=0)

        corr_sum_knowl_conc = []
        corr_sum_knowl_cog_cap = []
        for timepoint in range(1000):
            # This line gives a RuntimeWarning because there are a lot of zeros in this array, this can be ignored.
            corr_conc = np.corrcoef(learned_skills_per_timepoint_matrix[timepoint, :],
                                    self.simulation.concentration_matrix[timepoint, :])[0, 1]
            corr_sum_knowl_conc.append(corr_conc)
            corr_cog_cap = np.corrcoef(learned_skills_per_timepoint_matrix[timepoint, :],
                                       self.simulation.cog_cap_matrix[timepoint, :])[0, 1]
            corr_sum_knowl_cog_cap.append(corr_cog_cap)

        axs[place].plot(corr_sum_knowl_conc, label='Concentration')
        axs[place].plot(corr_sum_knowl_cog_cap, label='Cog cap')
        axs[place].set_title(str(plot_nr) + ": Correlation With Learned Skills")
        axs[place].set_xlabel('Timepoints')
        axs[place].set_ylabel('Correlation')
        axs[place].legend()

    def plot_test_structure(self, axs, place, plot_nr):
        """ Plot with required cognitive capacity and average cognitive capacity for m in test matrix """

        all_tests = np.concatenate((self.simulation.test_matrix[0:100, 0], self.simulation.test_matrix[0:100, 0],
                                    self.simulation.test_matrix[100:, 0], self.simulation.test_matrix[100:, 0]))

        axs[place].plot(np.arange(all_tests.size), all_tests,
                        label='Required Cognitive Capacity')

        for person in range(3):
            cog_cap = self.simulation.get_cog_cap(person)
            test_timepoints = np.multiply((params.nrOfTestOccasions / params.TOTAL_YEARS_OF_SIMULATION),
                                          params.TEST_AGES)

            selection = []
            for timepoint in test_timepoints.astype(int):
                selection.extend(cog_cap[timepoint: int(timepoint + 100)])

            axs[place].plot(np.arange(all_tests.size), selection,
                            label='Person (p=' + str(person) + ') Cognitive Capacity')

        axs[place].set_xlabel('Microskills')
        axs[place].set_ylabel('Cognitive Capacity')
        axs[place].set_title(str(plot_nr) + ': Test Structure')
        axs[place].legend()

    def plot_corr_test_items(self, axs, place, test_nr, plot_nr):
        """
        Plot correlation matrix of item groups
        """

        df = self.prepare_data(test_nr)

        axs[place].matshow(df.corr(), cmap='coolwarm')
        axs[place].set_title(str(plot_nr) + ". Corr between item groups",
                             fontdict={'fontsize': 10})
        axs[place].set_xlabel('Item group')
        axs[place].set_ylabel('Item group')

    def plot_raw_iq_scores(self, axs, place, title, test_start, test_end, plot_nr):
        """ Frequency histogram of number nr of people that passed nr of items """

        start_p = int(0)
        last_p = params.nrOfPersInTest
        IQ_test_selected_persons = self.simulation.learning_matrix[test_start:test_end,
                                   start_p:last_p]  # microskill, person
        raw_iq_scores = IQ_test_selected_persons.sum(axis=0)

        axs[place].hist(raw_iq_scores, len(raw_iq_scores))
        axs[place].set_title(str(plot_nr) + title, fontdict={'fontsize': 10})
        axs[place].set_xlabel('Nr of Items Passed')
        axs[place].set_ylabel('Nr of people)')

    def plot_corr_iq_scores(self, axs, place, characteristic, test_nr, plot_nr):
        """ Check correlation between iq score and concentration/cognitive capacity """

        start_p = int(0)
        last_p = params.nrOfPersInTest

        last_item = int(params.nrOfTestOccasions + (test_nr * 100))

        IQ_test_selected_persons = self.simulation.learning_matrix[last_item - 100:last_item,
                                   start_p:last_p]  # microskill, person
        raw_iq_scores = IQ_test_selected_persons.sum(axis=0)

        if characteristic == 'cog_cap':
            char = self.simulation.personality_matrix[start_p:last_p, 0]
            corr = np.corrcoef(raw_iq_scores, char)[0, 1]
            axs[place].set_ylabel('Cognitive Capacity')
            axs[place].set_title(str(plot_nr) + ': Corr IQ score and Cog Cap: ' + str(round(corr, 2)),
                                 fontdict={'fontsize': 10})

        if characteristic == 'concentration':
            char = self.simulation.personality_matrix[start_p:last_p, 1]
            corr = np.corrcoef(raw_iq_scores, char)[0, 1]
            axs[place].set_ylabel('Concentration')
            axs[place].set_title(str(plot_nr) + ': Corr IQ Score and Conc: ' + str(round(corr, 2)),
                                 fontdict={'fontsize': 10})

        fit = np.polyfit(raw_iq_scores, char, deg=1)  # add regression line to plot
        axs[place].plot(raw_iq_scores, char, 'bo')
        axs[place].plot(raw_iq_scores, fit[0] * raw_iq_scores + fit[1])
        axs[place].set_xlabel('Raw IQ Scores')

    def plot_cog_cap_structure(self):
        """ Plot cognitive capacity to get insight in similarity between different twin types"""

        for person in range(10):
            cog_cap = self.simulation.get_cog_cap(person)
            plt.plot(np.arange(params.nrOfMicroskillsNormal), cog_cap,
                            label='Person (p=' + str(person) + ') Cognitive Capacity')

        plt.title(params.PERS_TWIN)
        plt.ylabel('Cognitive Capacity')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    def factor_analysis(self, testnr):
        """ Perform factor analysis on iq test, store results in Docs/Test_testnr"""

        df = self.prepare_data(testnr)

        # Bartlett’s test of sphericity, checks whether or not the observed variables intercorrelate
        chi_square_value, p_value = calculate_bartlett_sphericity(df)

        # Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis
        kmo_all, kmo_model = calculate_kmo(df)
        plt.figtext(0.01, 0.8 + (testnr / 10), "Bartlett Sphericity: " +
                    str(round(chi_square_value, 2)) +
                    ", P-value: " + str(round(p_value, 2)) +
                    "\nkmo_model: " + str(round(kmo_model, 2)))

        # Create factor analysis object and perform factor analysis
        fa = FactorAnalyzer(n_factors=10, rotation="varimax")
        fa.fit(df)

        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()

        # Create scree plot using matplotlib
        # axs[place].scatter(range(1, df.shape[1] + 1), ev)
        # axs[place].plot(range(1, df.shape[1] + 1), ev)
        # axs[place].set_title('Scree Plot')
        # axs[place].set_xlabel('Factors')
        # axs[place].set_ylabel('Eigenvalue')
        # axs[place].grid()

        # Create factor analysis object and perform factor analysis
        pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6', 'Factor7',
                                            'Factor8', 'Factor9', 'Factor10', ]).to_csv("Docs/Test_" + str(testnr) +
                                                                                        "/fa_loadings.csv")

        # Get variance of each factors
        pd.DataFrame(fa.get_factor_variance(),
                     index=['SS Loadings', 'Proportion Var', 'Cumulative Var']).to_csv("Docs/Test_" + str(testnr) +
                                                                                       "/factor_variance.csv")

    def prepare_data(self, testnr):
        """ Prepare data for factor analysis or plot. Extracts the data for the right test """

        last_index = int(params.nrOfTestOccasions + (testnr * 100))

        test_scores = self.simulation.learning_matrix[(last_index - 100):last_index, :]
        scores_by_similar_items = np.zeros((10, 100))
        select_items = np.arange(0, 100, 10)

        for var in range(10):
            scores_by_similar_items[var, :] = test_scores[select_items, :].sum(axis=0)
            select_items = select_items + 1

        df = pd.DataFrame(scores_by_similar_items.T, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        df.to_csv("Docs/Test_" + str(testnr) + "/input_data_fa.csv")

        return df
