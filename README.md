# Ground Truth

Ground Truth Analysis Tooling

**This repository contains software that is not officially supported by Indico. It may
  be outdated or contain bugs. The operations it performs are potentially destructive.
  Use at your own risk.**

Requires Python ^3.10 and UV ^0.11

``` shell
$ uv tool install https://github.com/IndicoDelivery/groundtruth.git
$ groundtruth --help
```

**CAVEATS:**
- This program requires results to be in file version 3 format.
- This program requires results to be from a workflow on IPA 7.2 or later.
- This program requires Auto Review to be enabled for the workflow.

## Analysis Process

`groundtruth` is used to aid the ground truth analysis process for a model or workflow.
This is the process of comparing model predictions against known ground truths at
various confidence thresholds to establish accuracy, volume, and STP performance
metrics.

The process starts with a folder of documents that will be used to establish ground
truth--in this example we'll call it `documents`. These documents will be uploaded to
the workflow using the `submit` command. The submission IDs, file names, and direct
review URLs of the submitted documents will be written to a CSV for future reference.

~~~ shell
groundtruth submit \
    --host try.indico.io --token indico_api_token.txt \
    --workflow-id 1234 --documents-folder documents \
    --submission-ids-file submission_ids.csv
~~~

Once all submissions have been processed and auto reviewed, download the auto reviewed
result files with the `retrieve` command. These contain the predictions to be compared
against ground truth later.

~~~ shell
groundtruth retrieve \
    --host try.indico.io --token indico_api_token.txt \
    --submission-ids-file submission_ids.csv \
    --results-folder auto_reviews
~~~

Failed submissions will be logged and may be retried with the `retry` command.

~~~ shell
groundtruth retry \
    --host try.indico.io --token indico_api_token.txt \
    --submission-ids-file submission_ids.csv
~~~

Now review all submissions in Indico to capture the ground truth "answer key." You may
use the review URLs in `submissions_ids.csv` to navigate to them directly.

Once all submissions have been reviewed, download the HITL reviewed result files. These
contain the ground truths to be compared.

~~~ shell
groundtruth retrieve \
    --host try.indico.io --token indico_api_token.txt \
    --submission-ids-file submission_ids.csv \
    --results-folder ground_truths
~~~

Extract the labels, values, and confidences for `auto_reviews` and `ground_truths` with
the `extract` command. Combine the output CSVs with the `combine` command. This will
automatically match predictions and ground truths by similarity.

~~~ shell
groundtruth extract \
    --results-folder auto_reviews \
    --samples-file auto_reviews.csv
~~~

~~~ shell
groundtruth extract \
    --results-folder ground_truths \
    --samples-file ground_truths.csv
~~~

~~~ shell
groundtruth combine \
    --ground-truths-file ground_truths.csv \
    --predictions-file auto_reviews.csv \
    --combined-file samples.csv
~~~

At this point, the samples in `samples.csv` should be manually reviewed to set the
`accurate` column to `TRUE` for any ground truth/prediction pairs that should be
considered accurate but whose values were not character-for-character identical. The
`edit_distance` and `similarity` columns can be used to bubble up to the top of the
samples file values that are likely to be accurate. Any corrections to the selected
ground truth values should also be made in the `ground_truth` column to be used for
future rounds of analysis.

After manual review and correction, the samples file can be analyzed using the `analyze`
command to produce accuracy, volume, and STP performance metrics for a range of
specified confidence thresholds. Any samples that should not be included in the
analysis (such as ground truths with no value) should be filtered out of the
samples file prior to analyzing it.

~~~ shell
groundtruth analyze \
    --samples-file samples.csv \
    --analysis-file analysis.csv \
    0.85 0.95 0.99 0.99999
~~~

Additional rounds of analysis can be performed after model remediation or auto review
enhancements have been made to determine the performance impact. Use the `submit`,
`retrieve`, and `extract` commands to process the same folder of documents through the
updated workflow, saving the results and IDs as a new CSV.

~~~ shell
groundtruth submit \
    --host try.indico.io --token indico_api_token.txt \
    --workflow-id 1234 --documents-folder documents \
    --submission-ids-file remediated_submission_ids.csv
~~~

~~~ shell
groundtruth retrieve \
    --host try.indico.io --token indico_api_token.txt \
    --submission-ids-file remediated_submission_ids.csv \
    --results-folder remediated_reviews
~~~

~~~ shell
groundtruth extract \
    --results-folder remediated_reviews \
    --samples-file remediations.csv
~~~

Use the `combine` command to combine the ground truths from the original round of
analysis with the remediated predictions from this round. Ground truths and predictions
will be matched up by document filename and field.

~~~ shell
groundtruth combine \
    --ground-truths-file samples.csv \
    --predictions-file remediations.csv \
    --combined-file remediated_samples.csv
~~~

Use the `analyze` command on the remediated samples file to calculate the remediated
performance metrics. This process can be repeated for as many rounds of remediation as
necessary.

Additional ground truth documents can be added to the set by submitting, reviewing,
retrieving, and extracting them using a separate submission IDs CSV and samples
file. Afterwards, the submission IDs and samples for the new documents can be merged
into the original samples file.
