# GPT_PPO 
# Applying PPO on the Attention Mechanism of the GPT-2 model

# Version 1
# This code aims to fine-tune a GPT-2 model for a custom text generation task using the Proximal Policy Optimization (PPO) algorithm, which is a reinforcement learning approach. Here's a detailed theoretical explanation of the code:

1- Importing necessary libraries: The code imports the required PyTorch, transformers, and numpy libraries to build the custom environment, model, and training classes.

2- TextGenerationEnvironment: This class creates an environment for the text generation task, where the agent interacts to generate text. It initializes the GPT-2 tokenizer and model, and defines methods to generate text, reset the environment, and take a step (i.e., append a token to the current text).

3- ModifiedGPT: This class extends the GPT-2 model to work with reinforcement learning. It adds an additional linear layer (action_layer) and softmax activation function to produce action probabilities for the tokens in the vocabulary. The forward method computes these action probabilities and, if provided, the cross-entropy loss using the ground-truth labels.

4- RewardFunction: This class defines a reward function based on the negative perplexity of the generated text. It takes a reference text and initializes the GPT-2 tokenizer and model. The __call__ method calculates the perplexity of the generated text using the GPT-2 model and returns the negative perplexity as the reward.

5- RolloutStorage: This class stores the rollout data (observations, actions, action probabilities, rewards, and dones) during the agent's interaction with the environment. It also computes the returns and advantages, which are used for updating the policy during training.

6- PPOTrainer: This class trains the agent using the PPO algorithm. It initializes the environment, model, reward function, and other hyperparameters. The train method collects rollout data by having the agent interact with the environment and then updates the policy using multiple epochs of minibatch updates.

7- main() function: The main function initializes the environment, model, reward function, and trainer. It then trains the agent for a specified number of steps and saves the trained model's weights.

8- generate_text() function: This function takes a text prompt, model, tokenizer, and desired text length and generates text using the trained model. It tokenizes the input prompt and passes it to the model to generate the output text.

9- Script execution: The script, when executed, runs the main function to train the model and then uses the generate_text function to generate text using the fine-tuned model.

# The main idea behind this code is to fine-tune a GPT-2 model for a custom text generation task using reinforcement learning, specifically the PPO algorithm. The agent learns to generate text similar to a given reference text by interacting with the custom environment and receiving feedback in the form of rewards from the custom reward function.



# Version 2 
# in this version i tried to enhance the reward function by designing a custom reward function. The main differences between the RewardFunction from the previous code (V1) and the CustomRewardFunction (V2) from the current code are: :

1- In the CustomRewardFunction, the **spacy library** is used to load an English language model (en_core_web_md), and the reference_text is converted into a spacy document object (self.reference_doc). This is not present in the RewardFunction.

2- The CustomRewardFunction has an additional max_length parameter in its constructor. This is used to apply a penalty if the generated text exceeds the maximum allowed length.

3- In the CustomRewardFunction, after calculating the perplexity-based reward, the code calculates the semantic similarity between the generated text and the reference text using the spacy document objects. This similarity is then added to the reward. This step is not present in the RewardFunction.

4- In the CustomRewardFunction, a penalty of 1 is applied to the reward if the generated text exceeds the maximum allowed length. This is not present in the RewardFunction.

# So, the CustomRewardFunction has additional components to take into account the semantic similarity between the generated text and the reference text, as well as a penalty for exceeding the maximum allowed length, while the RewardFunction only considers perplexity-based rewards.



# The choice to use SpaCy instead of the GPT model for calculating the semantic similarity in the reward function is mainly due to the differences in the approaches of these two tools :

1- SpaCy is specifically designed for natural language processing tasks, and it comes with pre-trained word embeddings that can effectively capture semantic information in text. The similarity between two text documents in SpaCy is calculated based on the similarity between their word embeddings, which can be a straightforward and computationally efficient way to estimate the semantic similarity between two pieces of text.

2- On the other hand, GPT models are primarily designed for language generation tasks. Although they can be fine-tuned to perform other NLP tasks, calculating the semantic similarity between two pieces of text is not one of their native capabilities. Using a GPT model for this purpose would likely require additional fine-tuning and adaptation, which could be more complex and computationally expensive compared to using SpaCy's built-in similarity function.

3- By using SpaCy, the code leverages an efficient and straightforward approach to estimating semantic similarity, without the need for additional fine-tuning or complex adaptation of the GPT model.




