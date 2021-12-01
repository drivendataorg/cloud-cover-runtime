#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cd /codeexecution

    echo "List installed packages"
    echo "######################################"
    conda list -n condaenv
    echo "######################################"

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./
    ls -alh

    if [ -z "$(find data/test_features -type f -iname '*.tif'| head -1)" ]
    then
        echo "ERROR: No input files detected. If you are testing the container with 'make test-submission', are there files in 'runtime/data/test_features'?"
        exit_code=1
    else
        if [ -f "main.py" ]
        then
            echo "Running code submission with Python"
            conda run --no-capture-output -n condaenv python main.py

	    # Test that submission is valid
	    echo "Testing that submission is valid"
	    conda run -n condaenv pytest -v tests/test_submission.py

	    echo "Compressing files in a gzipped tar archive for submission"
	    cd ./predictions \
	      && tar czf ./submission.tar.gz *.tif \
	      && rm ./*.tif \
	      && cd ..

	    mv predictions/submission.tar.gz submission

	    echo "... finished"
	    du -h submission/submission.tar.gz

        else
            echo "ERROR: Could not find main.py in submission.zip"
            exit_code=1
        fi
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

cp /codeexecution/submission/log.txt /tmp/log
echo $exit_code
exit $exit_code
