from stage1_sctsr_v4.source_identity import build_source_tree_manifest

def test_source_manifest_hashes_registered_files(tmp_path):
    (tmp_path/'a').mkdir();(tmp_path/'a'/'x.py').write_text('x=1\n')
    m=build_source_tree_manifest(tmp_path,['a']);assert len(m['files'])==1;assert len(m['source_tree_digest'])==64
