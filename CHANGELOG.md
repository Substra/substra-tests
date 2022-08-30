# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- chore: rename connect-tools to substra-tools

## [0.31.0] - 2022-08-29

### Removed

- local folder tests

### Changed

- chore: change the assets and tools algo to feed with inputs outputs dictionary. (#256)

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

- Drop Python 3.7 support (#318)

### Added

- Add expiration date to skipped tests

## [0.27.0] - 2022-08-01

### Changed

- BREAKING: Add "inputs" field to compute task specs (#306)

## [0.26.0] - 2022-07-11

### Changed

- BREAKING: Add "outputs" field to compute task specs (#290)

## [0.25.0] - 2022-07-11

### Changed

- BREAKING: convert (test task) to (predict task + test task)

## [0.24.0] - 2022-07-05

### Changed

- Stop using metrics APIs, use algo APIs instead (#286)
- Following the change in substra, rename `Client.update_compute_plan` in `Client.add_compute_plan_tuples` (#294)

## [0.23.0] - 2022-06-27

### Changed

- feat: use new filters implementation for list functions (#275)
- remove `test-ci` command, the ci run only the remote tests (#288)

## [0.22.0] - 2022-06-14

### Added

- workflow: configurable number of datasamples to use for Mnist (#264)

### Changed

- feat: rename node to organization (#282)

## [0.21.0] - 2022-06-07

### Added

- wait for tuple error type (#274)

## [0.20.0] - 2022-05-30

### Added

- feat: add slack bot for sdk local tests (publish on #component_substra_sdk with the Substra-CI app) (#270)

### Changed

- feat: in subprocess mode, errors are now caught as `ExecutionError` (#271)

## [0.19.0] - 2022-05-23

### Added

- Test if the GPU is seen from the algo (#248)

## [0.18.0] - 2022-05-16

### Changed

- BREAKING CHANGE: add mandatory name field to compute plan (#257)

## [0.17.0] - 2022-05-09

### Added

- BREAKING CHANGE: feat!: update tests to give a key to ComputePlanSpec
- Add a test on GPU support, skipped by default (#248)

## [0.16.0] - 2022-05-03

### Added

- test: get_performances from a compute plan key (#243)
- test: test the base docker image with kaniko #245

### Changed

- chore: refactor config (#246)

### Fixed

- fix: add missing type hint import (#247)

## [0.15.0] - 2022-04-19

### Changed

- feat: fully covering debug mode test (#230)

### Fixed

- fix: set remote tag to latest (#241)
- fix: permissions head model test (#240)

## [0.14.0] - 2022-04-11

### Added

- ci: option to chose orchestrator mode (#170)

### Changed

- ci: metric endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls (#206)
- ci: algo endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls (#207)
- ci: datamanager endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls (#209)
- ci: tuple endpoints now returns 404 instead of 400 for non-uuid keys in retrieve calls (#216)
- ci: compute plan endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls (#219)
- feat: use latest connect-tools image by default (#237)

## [0.13.0] - 2022-03-01

### Added

- feat: add support for Connect GitHub Bot (#175)
- ci: workflow dispatch - be able to deactivate the sdk tests (#165)
- chore(ci): enable/disable frontend/substrafl tests when triggering the workflow manually (#154)

### Changed

- ci: the main branches of the substra and connect-tools repositories are now main (#143 #144 #146)
- The base Docker image is connect-tools 0.10.0, with Python 3.9

## [0.12.0] - 2022-01-16

### Added

- implement the substrafl e2e tests (#126)
- ci: export events logs (#136)
- add logs permission to dataset (#130)
- ci: Be able to run only the substrafl tests (#165 #173 #179)

### Changed

- increase coverage of algo.list view (#137)
- ci: the main branches of the substra and connect-tools repositories are now main (#143 #144 #146)

### Fixed

- ci: fix links in slack messages (#138)

## [0.11.0] - 2022-01-10

### Added

- add frontend testing, improve slack reporting (#53)
- test error type in tuple api responses (#119)

### Changed

- algo writes in the home directory (#111)
- format code with Black and isort
- update acceptable concommit subject pattern (#122)
- more compact Slack reports + add e2e tests duration (#128)

### Fixed

- GHA workflow slack reporting condition (#121)
- handle absent test duration in slack reporting (#131)

## [0.10.0] - 2021-12-01

### Added

- test on algo download (#109)

### Changed

- update tests for compatibility with backend (#107)
- improve tests coverage on task compatibility (#92)

## [0.9.0] - 2021-11-02

### Added

- test returned task counts with cp (#91)
- add default values for user inputs (#99)
- test start end date integration (#93)
- use kaniko prepolutaed images (#95)
- add echo step of workflow inputs (#90)
- make repo branches configurable for e2e tests in github action (#86)
- enforce conventional PR titles (#84)
- multiple metrics per testtask (#74)

### Changed

- rename tests.test_debug module (#103)
- improve code for local images (#102)
- update skaffold version (#80)
- update the webhook to target a new channel (#81)
- Retry a bit when images fail building (#79)

### Fixed

- handle canceled and failed cp in client.wait method (#100)
- task registration on a disabled model should fail (#97)
- remove the random id generator that fails everyday (#98)
- remove a pipe to avoid a broken pipe error (#94)
- missing inputs field (#89)
- fix workflow indentation (#88)
- add helm couchdb repo (#85)
- do not fail if output adds extra lines (#83)
- update metric and testtuple add for the mnist workflow (#82)
- access the local folder path properly instead of using a hardcoded path (#76)

## [0.8.0] - 2021-10-04

### Added

- Option to `run-ci.py` to create a cluster with multiple kubernetes nodes (#37)
- Options `--no-cleanup` and `--use-cluster` to `run-ci.py` (#39)
- `cleanup.py` in addition to `run-ci.py` (#49)
- Orchestrator (#2)
- Run nightly tests in standalone and distributed mode (#63)
- Twuni helm repo (#65)

### Changed

- Unify all algos into one (#61)
- Testtuple after aggregate tuple (#32)

### Removed

### Fixed

## [0.7.0] - 2021-08-04

## [0.6.2] - 2021-06-28

## [0.6.1] - 2021-04-13
