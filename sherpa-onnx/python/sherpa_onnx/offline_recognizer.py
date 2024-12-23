# Copyright (c)  2023 by manyeyes
# Copyright (c)  2023  Xiaomi Corporation
from pathlib import Path
from typing import List, Optional

from _sherpa_onnx import (
    FeatureExtractorConfig,
    OfflineCtcFstDecoderConfig,
    OfflineLMConfig,
    OfflineModelConfig,
    OfflineMoonshineModelConfig,
    OfflineNemoEncDecCtcModelConfig,
    OfflineParaformerModelConfig,
)
from _sherpa_onnx import OfflineRecognizer as _Recognizer
from _sherpa_onnx import (
    OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
    OfflineStream,
    OfflineTdnnModelConfig,
    OfflineTransducerModelConfig,
    OfflineWenetCtcModelConfig,
    OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
)


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OfflineRecognizer(object):
    """A class for offline speech recognition.

    Please refer to the following files for usages
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/tests/test_offline_recognizer.py
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py
    """

    @classmethod
    def from_transducer(
        cls,
        encoder: str,
        decoder: str,
        joiner: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        blank_penalty: float = 0.0,
        modeling_unit: str = "cjkchar",
        bpe_vocab: str = "",
        debug: bool = False,
        provider: str = "cpu",
        model_type: str = "transducer",
        rule_fsts: str = "",
        rule_fars: str = "",
        lm: str = "",
        lm_scale: float = 0.1,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          encoder:
            Path to ``encoder.onnx``.
          decoder:
            Path to ``decoder.onnx``.
          joiner:
            Path to ``joiner.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values: greedy_search, modified_beam_search.
          max_active_paths:
            Maximum number of active paths to keep. Used only when
            decoding_method is modified_beam_search.
          hotwords_file:
            The file containing hotwords, one words/phrases per line, and for each
            phrase the bpe/cjkchar are separated by a space.
          hotwords_score:
            The hotword score of each token for biasing word/phrase. Used only if
            hotwords_file is given with modified_beam_search as decoding method.
          blank_penalty:
            The penalty applied on blank symbol during decoding.
          modeling_unit:
            The modeling unit of the model, commonly used units are bpe, cjkchar,
            cjkchar+bpe, etc. Currently, it is needed only when hotwords are
            provided, we need it to encode the hotwords into token sequence.
            and the modeling unit is bpe or cjkchar+bpe.
          bpe_vocab:
            The vocabulary generated by google's sentencepiece program.
            It is a file has two columns, one is the token, the other is
            the log probability, you can get it from the directory where
            your bpe model is generated. Only used when hotwords provided
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            transducer=OfflineTransducerModelConfig(
                encoder_filename=encoder,
                decoder_filename=decoder,
                joiner_filename=joiner,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            modeling_unit=modeling_unit,
            bpe_vocab=bpe_vocab,
            model_type=model_type,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        if len(hotwords_file) > 0 and decoding_method != "modified_beam_search":
            raise ValueError(
                "Please use --decoding-method=modified_beam_search when using "
                f"--hotwords-file. Currently given: {decoding_method}"
            )

        if lm and decoding_method != "modified_beam_search":
            raise ValueError(
                "Please use --decoding-method=modified_beam_search when using "
                f"--lm. Currently given: {decoding_method}"
            )

        lm_config = OfflineLMConfig(
            model=lm,
            scale=lm_scale,
            lm_num_threads=num_threads,
            lm_provider=provider,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            lm_config=lm_config,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
            hotwords_file=hotwords_file,
            hotwords_score=hotwords_score,
            blank_penalty=blank_penalty,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_sense_voice(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        language: str = "",
        use_itn: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_
        to download pre-trained models.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          model:
            Path to ``model.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          language:
            If not empty, then valid values are: auto, zh, en, ja, ko, yue
          use_itn:
            True to enable inverse text normalization; False to disable it.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            sense_voice=OfflineSenseVoiceModelConfig(
                model=model,
                language=language,
                use_itn=use_itn,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_paraformer(
        cls,
        paraformer: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html>`_
        to download pre-trained models.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          paraformer:
            Path to ``model.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            paraformer=OfflineParaformerModelConfig(model=paraformer),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="paraformer",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_telespeech_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 40,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_
        to download pre-trained models.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model. It is
            ignored and is hard-coded in C++ to 40.
          feature_dim:
            Dimension of the feature used to train the model. It is ignored
            and is hard-coded in C++ to 40.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            telespeech_ctc=model,
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="nemo_ctc",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_nemo_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="nemo_ctc",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_whisper(
        cls,
        encoder: str,
        decoder: str,
        tokens: str,
        language: str = "en",
        task: str = "transcribe",
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        tail_paddings: int = -1,
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html>`_
        to download pre-trained models for different kinds of whisper models,
        e.g., tiny, tiny.en, base, base.en, etc.

        Args:
          encoder:
            Path to the encoder model, e.g., tiny-encoder.onnx,
            tiny-encoder.int8.onnx, tiny-encoder.ort, etc.
          decoder:
            Path to the decoder model, e.g., tiny-decoder.onnx,
            tiny-decoder.int8.onnx, tiny-decoder.ort, etc.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          language:
            The spoken language in the audio file. Example values: en, de, zh,
            jp, fr. See https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
            for all possible values. Note that for non-multilingual models, the
            only valid value is 'en'.
          task:
            Valid values are: transcribe, translate. Note that for
            non-multilingual models, the only valid value is 'transcribe'.
          num_threads:
            Number of threads for neural network computation.
          decoding_method:
            Valid values: greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            whisper=OfflineWhisperModelConfig(
                encoder=encoder,
                decoder=decoder,
                language=language,
                task=task,
                tail_paddings=tail_paddings,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="whisper",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=16000,
            feature_dim=80,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_moonshine(
        cls,
        preprocessor: str,
        encoder: str,
        uncached_decoder: str,
        cached_decoder: str,
        tokens: str,
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html>`_
        to download pre-trained models for different kinds of moonshine models,
        e.g., tiny, base, etc.

        Args:
          preprocessor:
            Path to the preprocessor model, e.g., preprocess.onnx
          encoder:
            Path to the encoder model, e.g., encode.int8.onnx
          uncached_decoder:
            Path to the uncached decoder model, e.g., uncached_decode.int8.onnx,
          cached_decoder:
            Path to the cached decoder model, e.g., cached_decode.int8.onnx,
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          decoding_method:
            Valid values: greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            moonshine=OfflineMoonshineModelConfig(
                preprocessor=preprocessor,
                encoder=encoder,
                uncached_decoder=uncached_decoder,
                cached_decoder=cached_decoder,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
        )

        unused_feat_config = FeatureExtractorConfig(
            sampling_rate=16000,
            feature_dim=80,
        )

        recognizer_config = OfflineRecognizerConfig(
            model_config=model_config,
            feat_config=unused_feat_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_tdnn_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 8000,
        feature_dim: int = 23,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html>`_
        to download pre-trained models.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            tdnn=OfflineTdnnModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="tdnn",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_wenet_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        rule_fsts: str = "",
        rule_fars: str = "",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            wenet_ctc=OfflineWenetCtcModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="wenet_ctc",
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    def create_stream(self, hotwords: Optional[str] = None):
        if hotwords is None:
            return self.recognizer.create_stream()
        else:
            return self.recognizer.create_stream(hotwords)

    def decode_stream(self, s: OfflineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OfflineStream]):
        self.recognizer.decode_streams(ss)
