# Ground Truth

Ground Truth Analysis Tooling

Requires Python ^3.9 and Poetry ^1.2

``` shell
$ poetry install
$ poetry run poe {format,check,test,all}
$ poetry run groundtruth --help
```

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
    --submission-ids-file ground_truth_submission_ids.csv
~~~

At this point, all of the submissions listed in the `ground_truth_submission_ids.csv`
will be queued for processing. Once processed, they should all be manually reviewed to
establish ground truth. After all reviews have been completed, the results can be
downloaded using the `retrieve` command.

~~~ shell
groundtruth retrieve \
    --host try.indico.io --token indico_api_token.txt \
    --submission-ids-file ground_truth_submission_ids.csv \
    --results-folder ground_truth_results
~~~

These results contain the ground truths *and* predictions for the first round of ground
truth analysis. Ground truths and prediction samples for a specific model can be
extracted from the results using the `extract` command.

~~~ shell
groundtruth extract \
    --results-folder ground_truth_results \
    --extractions-file ground_truth_extractions.csv \
    --model "Invoice Extraction Model"
~~~

At this point, the ground truth/prediction samples in `ground_truth_extractions.csv`
should be manually reviewed to set the `accurate` column to `TRUE` for any samples that
should be considered accurate but that were not identical. The `edit_distance` and
`similarity` columns can be used to bubble up to the top of the extractions file
samples that are likely to be accurate. Any corrections to the selected ground truth
values should also be made in the `ground_truth` column to be used for this and future
rounds of analysis.

After manual review and correction, the extractions file can be analyzed using the
`analyze` command to produce accuracy, volume, and STP performance metrics for a range
of confidence thresholds. Any samples that should not be included in the analysis
(such as ground truths with no value) should be filtered out of the extractions file
prior to analyzing it.

~~~ shell
groundtruth analyze \
    --extractions-file ground_truth_extractions.csv \
    --analysis-file ground_truth_analysis.csv \
    0.85 0.95 0.99 0.99999
~~~

Additional rounds of analysis can be performed after model remediation or auto review
enhancements have been made to determine the performance impact. Use the `submit`,
`retrieve`, and `extract` commands to process the same folder of documents through the
updated workflow, saving the results and IDs as a new set.

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
    --results-folder remediated_results
~~~

~~~ shell
groundtruth extract \
    --results-folder remediated_results \
    --extractions-file remediated_extractions.csv \
    --model "Invoice Extraction Model"
~~~

Note that the results and extractions will *not* contain ground truths, only remediated
predictions. Use the `combine` command to combine the ground truths from the original
round of analysis with the remediated predictions from this round. Ground truths and
predictions will be matched up by document file name and field.

~~~ shell
groundtruth combine \
    --ground-truths-file ground_truth_extractions.csv \
    --predictions-file remediated_extractions.csv \
    --extractions-file combined_extractions.csv
~~~

Use the `analyze` command on the combined extractions CSV to calculate the remediated
performance metrics. This process can be repeated for as many rounds of remediation as
necessary.

Additional ground truth documents can be added to the set by submitting, reviewing,
retrieving, and extracting them using a separate submission IDs CSV and extractions
CSV. Afterwards, the submission IDs and extractions for the new documents can be merged
into the original CSVs.
