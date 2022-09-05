# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- fix(tests): safe getter for optional inputs
- chore: change the Dockerfile `ENTRYPOINT` definition of algorithm relying on `substratools` to pass the method to execute under the `--method-name` argument
- chore: rename connect-tools to substra-tools
- tests: download tests optimized using the new substra feature: download function return the path to the downloaded file.

## [0.31.0] - 2022-08-29

### Removed

- local folder tests

### Changed

- chore: change the assets and tools algo to feed with inputs outputs dictionary. 

### Added

- A test for task profiling reports

### Fixed

- fix the test_base_connect_tools_image test.

## [0.30.0] - 2022-08-22

### Added

- Added a test for transient outputs

## [0.29.0] - 2022-08-17

### Fixed

- Fix the tests so that the task execution uses inputs/outputs

### Added

- Added tests for update name methods

## [0.28.0] - 2022-08-09

### Changed

- Drop Python 3.7 support 

### Added

- Add expiration date to skipped tests

## [0.27.0] - 2022-08-01

### Changed

- BREAKING: Add "inputs" field to compute task specs 

## [0.26.0] - 2022-07-11

### Changed

- BREAKING: Add "outputs" field to compute task specs 

## [0.25.0] - 2022-07-11

### Changed

- BREAKING: convert (test task) to (predict task + test task)

## [0.24.0] - 2022-07-05

### Changed

- Stop using metrics APIs, use algo APIs instead 
- Following the change in substra, rename `Client.update_compute_plan` in `Client.add_compute_plan_tuples` 

## [0.23.0] - 2022-06-27

### Changed

- feat: use new filters implementation for list functions 
- remove `test-ci` command, the ci run only the remote tests 

## [0.22.0] - 2022-06-14

### Added

- workflow: configurable number of datasamples to use for Mnist 

### Changed

- feat: rename node to organization 

## [0.21.0] - 2022-06-07

### Added

- wait for tuple error type 

## [0.20.0] - 2022-05-30

### Added

- feat: add slack bot for sdk local tests (publish on #component_substra_sdk with the Substra-CI app) 

### Changed

- feat: in subprocess mode, errors are now caught as `ExecutionError` 

## [0.19.0] - 2022-05-23

### Added

- Test if the GPU is seen from the algo 

## [0.18.0] - 2022-05-16

### Changed

- BREAKING CHANGE: add mandatory name field to compute plan 

## [0.17.0] - 2022-05-09

### Added

- BREAKING CHANGE: feat!: update tests to give a key to ComputePlanSpec
- Add a test on GPU support, skipped by default 

## [0.16.0] - 2022-05-03

### Added

- test: get_performances from a compute plan key 
- test: test the base docker image with kaniko #245

### Changed

- chore: refactor config 

### Fixed

- fix: add missing type hint import 

## [0.15.0] - 2022-04-19

### Changed

- feat: fully covering debug mode test 

### Fixed

- fix: set remote tag to latest 
- fix: permissions head model test 

## [0.14.0] - 2022-04-11

### Added

- ci: option to chose orchestrator mode 

### Changed

- ci: metric endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls 
- ci: algo endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls 
- ci: datamanager endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls 
- ci: tuple endpoints now returns 404 instead of 400 for non-uuid keys in retrieve calls 
- ci: compute plan endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls 
- feat: use latest connect-tools image by default 

## [0.13.0] - 2022-03-01

### Added

- feat: add support for Connect GitHub Bot 
- ci: workflow dispatch - be able to deactivate the sdk tests 
- chore(ci): enable/disable frontend/substrafl tests when triggering the workflow manually 

### Changed

- ci: the main branches of the substra and connect-tools repositories are now main (#143 #144 #146)
- The base Docker image is connect-tools 0.10.0, with Python 3.9

## [0.12.0] - 2022-01-16

### Added

- implement the substrafl e2e tests 
- ci: export events logs 
- add logs permission to dataset 
- ci: Be able to run only the substrafl tests (#165 #173 #179)

### Changed

- increase coverage of algo.list view 
- ci: the main branches of the substra and connect-tools repositories are now main (#143 #144 #146)

### Fixed

- ci: fix links in slack messages 

## [0.11.0] - 2022-01-10

### Added

- add frontend testing, improve slack reporting 
- test error type in tuple api responses 

### Changed

- algo writes in the home directory 
- format code with Black and isort
- update acceptable concommit subject pattern 
- more compact Slack reports + add e2e tests duration 

### Fixed

- GHA workflow slack reporting condition 
- handle absent test duration in slack reporting 

## [0.10.0] - 2021-12-01

### Added

- test on algo download 

### Changed

- update tests for compatibility with backend 
- improve tests coverage on task compatibility 

## [0.9.0] - 2021-11-02

### Added

- test returned task counts with cp 
- add default values for user inputs 
- test start end date integration 
- use kaniko prepolutaed images 
- add echo step of workflow inputs 
- make repo branches configurable for e2e tests in github action 
- enforce conventional PR titles 
- multiple metrics per testtask 

### Changed

- rename tests.test_debug module 
- improve code for local images 
- update skaffold version 
- update the webhook to target a new channel 
- Retry a bit when images fail building 

### Fixed

- handle canceled and failed cp in client.wait method 
- task registration on a disabled model should fail 
- remove the random id generator that fails everyday 
- remove a pipe to avoid a broken pipe error 
- missing inputs field 
- fix workflow indentation 
- add helm couchdb repo 
- do not fail if output adds extra lines 
- update metric and testtuple add for the mnist workflow 
- access the local folder path properly instead of using a hardcoded path 

## [0.8.0] - 2021-10-04

### Added

- Option to `run-ci.py` to create a cluster with multiple kubernetes nodes 
- Options `--no-cleanup` and `--use-cluster` to `run-ci.py` 
- `cleanup.py` in addition to `run-ci.py` 
- Orchestrator 
- Run nightly tests in standalone and distributed mode 
- Twuni helm repo 

### Changed

- Unify all algos into one 
- Testtuple after aggregate tuple 

### Removed

### Fixed

## [0.7.0] - 2021-08-04

## [0.6.2] - 2021-06-28

## [0.6.1] - 2021-04-13
