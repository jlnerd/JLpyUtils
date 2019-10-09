import pytest

import JLpyUtils

# def test_kaggle_setupe_config_dir():
#     JLpyUtils.kaggle.setup_config_dir(username='jtleona01', key = 'foo')

def test_kaggle_competition_download_files(tmpdir):
    
    try:
        JLpyUtils.kaggle.competition_download_files(competition='foo',
                                        path_report_dir=tmpdir)
    except Exception as e:
        assert('Reason: Unauthorized' in str(e) or 'Could not find kaggle.json' in str(e) or 'foo is not a valid competition' in str(e)), 'JLpyUtils.kaggle.competition_download_files is returning an unexpected error:'+str(e)