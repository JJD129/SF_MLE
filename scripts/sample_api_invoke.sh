# Note: This is a sample for a Mac. Please adjust as needed to your operating system
# The output file 'test_inference.json' will be included in your submission.

curl -X POST "http://127.0.0.1:8000/inference/" -H "accept: application/json" -H "Content-Type: application/json" -d @candidate_27_test_inference.json -o test_inference.json