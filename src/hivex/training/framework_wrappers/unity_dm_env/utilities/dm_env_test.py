# Copyright 2022 The HIVEX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HIVEX dm_env environment check."""

from absl.testing import absltest
from dm_env import test_utils
from hivex.training.unity_dm_env.unity_dm_env_wrapper import HivexDmEnvWrapper

from hivex.training.wrapper_utils import UnityEnvironment

ENV_PATH = (
    "./environments/test/hivex_test_env_rolling_ball_win/ML-Rolling-Ball_Unity.exe"
)

unity_env = UnityEnvironment(
    file_name=ENV_PATH,
    no_graphics=True,
)


class HivexTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return HivexDmEnvWrapper(unity_env=unity_env)


def check_unity_dm_environment():
    absltest.main()


if __name__ == "__main__":
    check_unity_dm_environment()
