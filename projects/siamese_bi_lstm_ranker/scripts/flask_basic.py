from flask import Flask, request, jsonify
from scripts.util import rank_passages_for_single_set_return_scores, single_set_return_attn
from scripts.util import read_word2_ind, load_models, read_ind2word, instantiate_and_load_models_new_flags
from scripts.util import return_ind_for_word_list, return_word_for_ind_list, write_lists_to_word_doc
from scripts.constant_model_params import params_to_load_models, device

app = Flask(__name__)


def build_app():
    app = Flask('latent')

    @app.route('/score', methods=['POST'])
    def get_input_and_score():
        payload = request.get_json()
        word2ind = read_word2_ind('common')
        model_name = 'model_bs_256_epoch_1_attn_biLSTM_201811011906'
        loaded_models = load_models(model_name, device, params_to_load_models, is_eval=True)
        q_encoder = loaded_models['question_encoder'].eval()
        p_encoder = loaded_models['passage_encoder'].eval()
        attn_model = loaded_models['attention_model']
        question = payload['question']
        list_of_passages = payload['answers']
        returned_scores = rank_passages_for_single_set_return_scores(question, list_of_passages,
                                                                     q_encoder, p_encoder,
                                                                     word2ind, device, attn_model)
        return jsonify(returned_scores)

    @app.route('/attend', methods=['POST'])
    def get_input_attend_and_store():
        payload = request.get_json()
        question = payload['question']
        answers = payload['answers']
        output_path = payload['outPath']

        word2ind = read_word2_ind('common')
        ind2word = read_ind2word('common')
        model_name = 'model_bs_256_epoch_140_attn_biLSTM_201811021840'
        new_parameters = params_to_load_models.copy()
        use_question_context = new_parameters.pop('use_question_context')
        return_context = new_parameters.pop('return_context')
        loaded_models_for_eval = instantiate_and_load_models_new_flags(model_name, device, use_question_context,
                                                                       return_context, new_parameters, is_eval=True)
        q_encoder = loaded_models_for_eval['question_encoder']
        p_encoder = loaded_models_for_eval['passage_encoder']
        attn_model = loaded_models_for_eval['attention_model']

        if not params_to_load_models['save_weights']:
            return "Please enable save_weights flag"

        question_attn, list_of_attn_passages = single_set_return_attn(question, answers, q_encoder, p_encoder, word2ind,
                                                             device, attn_model)

        list_of_word_lists = []
        list_of_attn= []
        ind_list_q = return_ind_for_word_list(question, word2ind)
        word_list_q = return_word_for_ind_list(ind_list_q, ind2word).split(' ')
        list_of_attn.append(question_attn)
        list_of_word_lists.append(word_list_q)
        for i, passage in enumerate(answers):
            ind_list = return_ind_for_word_list(passage, word2ind)
            word_list = return_word_for_ind_list(ind_list, ind2word).split(' ')
            list_of_word_lists.append(word_list)
        list_of_attn += list_of_attn_passages
        write_lists_to_word_doc(output_path, list_of_word_lists, list_of_attn)

        return "File saved please check {}".format(output_path)

    return app


app = build_app()

if __name__ == '__main__':
    app.run('0.0.0.0', port=5002)
