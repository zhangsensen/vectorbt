# CI Configuration

This directory contains GitHub Actions workflows for continuous integration.

## Workflows

### 1. `ci.yml` - Full CI Pipeline
- **Trigger**: Push to main/develop branches, PRs, daily schedule
- **Jobs**:
  - `smoke-test`: Runs the smoke test and limited full analysis
  - `dependency-check`: Checks for security vulnerabilities
  - `code-quality`: Runs code formatting and linting checks
  - `notification`: Sends failure notifications

### 2. `smoke-test.yml` - Dedicated Smoke Test
- **Trigger**: Push to main/develop branches, PRs, manual dispatch
- **Purpose**: Quick validation of core functionality
- **Features**:
  - Runs `python smoke_test.py`
  - Uploads logs as artifacts
  - Comments on PRs with results
  - Creates test summary

## Purpose

The CI pipeline is designed to:
1. **Prevent Regression**: Automatically catch timezone and timestamp handling issues
2. **Ensure Quality**: Maintain code standards and security
3. **Provide Feedback**: Quick feedback on code changes
4. **Document Issues**: Capture and report failures for debugging

## Smoke Test Coverage

The smoke test validates:
- ✅ Timezone handling (Asia/Hong_Kong)
- ✅ Data loading from parquet files
- ✅ Factor analysis (RSI, MACD, etc.)
- ✅ Memory management and cleanup
- ✅ Timestamp column preservation
- ✅ Multiprocessing data serialization
- ✅ JSON serialization of results
- ✅ WFO window processing

## Failure Prevention

Based on the critical issue fixed on 2025-09-09, the CI pipeline specifically tests for:
- Timezone comparison errors
- Unix timestamp conversion issues
- DatetimeIndex preservation in multiprocessing
- JSON serialization of numpy int64 types
- Data integrity across process boundaries

## Manual Testing

To run the smoke test locally:
```bash
python smoke_test.py
```

To trigger the CI pipeline manually:
1. Go to the Actions tab in GitHub
2. Select the workflow
3. Click "Run workflow"

## Notifications

The pipeline will:
- Comment on PRs with smoke test results
- Create GitHub step summaries
- Send failure notifications
- Upload logs for debugging