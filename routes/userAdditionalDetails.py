# from flask import Blueprint, request, jsonify
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Load model and tokenizer globally (only once)
# model_name = "t5-small"  # or "t5-base" for better results
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Create a Blueprint for routes
# user_bp = Blueprint("userAdditionalDetails", __name__)



# #overallfunction + router make same controller 
# def summarize_text(text, max_length_ratio=0.50):
#     """
#     Summarizes the input text using the T5 model.
#     The summary length is controlled by the max_length_ratio.
#     """
#     input_text = "summarize: " + text
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

#     input_tokens_count = inputs.shape[-1]
#     summary_max_length = max(10, int(input_tokens_count * max_length_ratio))  # prevent too short

#     summary_ids = model.generate(
#         inputs,
#         max_length=summary_max_length,
#         min_length=max(5, int(summary_max_length * 0.5)),
#         length_penalty=2.0,
#         num_beams=4,
#         early_stopping=True
#     )

#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# @user_bp.route("/userAdditionalDetails", methods=["POST"])
# def userDetails():
#     try:
#         data = request.get_json(force=True)  # force=True handles wrong headers
#         complain_text = data.get("complain_text", "").strip()

#         if not complain_text:
#             return jsonify({"error": "complain_text is required"}), 400

#         summarize_complain = summarize_text(complain_text)
#         return jsonify({"summary": summarize_complain})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


