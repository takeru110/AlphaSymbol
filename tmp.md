./src/model_meta/data.pyのPREDataModuleを以下のように書き換えて

- self.dfをpd.read_csvではなく、datasetsモジュールを利用してcsv_pathからデータセットを取り込んでself.datasetに格納する
    - 注意点：DataFrameの作成は時間がかかるので、datasetsによって効率的に必要な部分だけcsvから取り込むようにしたい
- build_vocab()を削除する。その代わりself.tgt_vocabはPREDataModule.__init__()で渡すtgt_vocab_listを用いて作成する。tgt_vocab_listは./datatraining/superfib_r1_metadata.yamlに書かれたようなものが来ると考える。また、同様にsrc_vocab_listからsrc_vocabも作成する。ただし双方のリストには"[BOS]", "[EOS]", "[PAD]", "[UNK]"を設ける。
- srcの加工方法は、datasetsの機能を利用して、self.src_add_ends()とself.src_pad_points()を施す。また、全ての要素に対してsrc_vocabを用いて変形を行う。
- point_vector_sizeは__init__()でmetadata.yamlより与えるmax_src_lentghに+2( bos, eosの3つ分)をした値にする。
- self.tgt_input_sizeはmetadata.yamlのmax_tgt_lengthの値に+2をした値にする。
- self.datasetの機能をよく利用してターゲットデータに対してself.tgt_add_endsとself.tgt_pad、tgt_vocabを施す。ただし、元のコードと処理結果が変わらないように注意。
- train_dataloader, val_dataloader, test_dataloaderに載せるデータはシャッフルされているようにすること