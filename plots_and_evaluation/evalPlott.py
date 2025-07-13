import pandas as pd
import seaborn as sns
import os
import glob
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt

def plot_scores(score: str,core_dir, path,label):
    """Plots a given score for all models found in CSV files."""
    result_dir = "core_dir,"
    csv_files = glob.glob(os.path.join(result_dir, "*.csv"))
    pca_dimensions = [0.8, 0.85, 0.9, 0.95]
 


    df = pd.read_csv(core_dir+path)

    
    def get_model_label(model_str):
        #remove occurence of string 'conf_' from model_str
        model_str = model_str.replace('conf_', '')
        architecture = model_str[:3]
        if model_str[-1] == 'a':
            return f'{architecture} no pca'
        else:
            return f'{architecture} pca= {str(pca_dimensions[int(model_str[-1])-1])}'

    df['model'] = df['model'].astype(str).apply(get_model_label)
    
    sns.set_style("whitegrid")
    #make font size bigger
    sns.set_context("notebook", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='model', y=score, marker='o', ci=None)
   

    #replace _ with sspace in y axis label
    plt.xlabel("Model")
    plt.ylabel(score.replace('_', ' ').title())
    #sns.barplot(data=df, x='model', y=score)
    score_label = score.replace('_', ' ').title()
    plt.title(f"Comparison of Models based on {score_label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()  # Commented out to avoid errors in headless environments
    # Use the directory from the input path for the output

    output_file = os.path.join(core_dir, f"{score}_comparison{label}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")


def final_test_scoresLoop(core_dir):
    label= "final_test_scores"
    scoreList = ['test_accuracy', 'test_ari','test_f1']
    for score in scoreList:
        plot_scores(score=score, core_dir=core_dir, path="final_test_scores.csv", label=label)

def main_pipeline_validationLoop(core_dir):
    label= "main_pipeline_validation_results"
    scoreList = ['validation_accuracy', 'validation_ari','validation_f1']
    for score in scoreList:
        plot_scores(score=score, core_dir=core_dir, path="main_pipeline_validation_results.csv", label=label)

def confidence_weighted_validation_resultsLoop(core_dir):
    label= "confidence_weighted_validation"
    scoreList = ['validation_accuracy', 'validation_ari','validation_f1']
    for score in scoreList:
        plot_scores(score=score, core_dir=core_dir, path="confidence_weighted_validation_results.csv", label=label)

def final_test_scores_confidence_weighted(core_dir):
    label= "final_test_scores_confidence_weighted"
    scoreList = ['test_accuracy', 'test_ari','test_f1']
    for score in scoreList:
        plot_scores(score=score, core_dir=core_dir, path="final_test_scores_confidence_weighted.csv", label=label)



#plot_scores(score='validation_accuracy',core_dir="results/leiden_0.3_adenocarcinoma_224" ,path="/main_pipeline_validation_results.csv",label="main_pipeline")
#plot_scores(score='validation_f1',path="main/results/leiden_0.3_adenocarcinoma_224/main_pipeline_validation_results.csv",label="0.3_adenocarcinoma_224")


#plot_scores(score='test_f1',core_dir=f"results_224/{subdir}/" ,path="final_test_scores.csv", label="final_test_scores")

# 224
# leiden_0.3 regular
subdir = "results_224/leiden_0.3_adenocarcinoma_224/"
final_test_scoresLoop(core_dir=subdir)
main_pipeline_validationLoop(core_dir=subdir)

# 224
# leiden_0.3 confidence
subdir = "results_224/leiden_0.3_adenocarcinoma_224_confidence/"
confidence_weighted_validation_resultsLoop(core_dir=subdir)
final_test_scores_confidence_weighted(core_dir=subdir)


# 448
# leiden_0.3 regular
subdir = "results_448/leiden_0.3_adenocarcinoma_448/"
final_test_scoresLoop(core_dir=subdir)
main_pipeline_validationLoop(core_dir=subdir)

# 448
# leiden_0.3 confidence
subdir = "results_448/leiden_0.3_adenocarcinoma_448_confidence/"
confidence_weighted_validation_resultsLoop(core_dir=subdir)
final_test_scores_confidence_weighted(core_dir=subdir)


