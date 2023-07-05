
# For Reviewer LTAA

## Comment：
We thank the reviewer for their detailed feedback. Please find detailed responses in the following.

>1.Why are there no ablation experiments for MADM? Is the Multi-Axis Dynamic Mix layer better than other feature extraction methods? 

In the ablation experiments, Table 3 specifically evaluates the impact of different feature extraction modules on MADM. We compare MADM with the baseline approach, which consists of a two-layer CNN. Additionally, we compare MADM with two novel feature extraction modules in image processing, namely ConvNeXt and HaloNet. As stated in the original text, "In Tab.3, different SRBs mean that the two layers of CNN in SRB [36] are replaced by ConvNeXt [18], HaloNet [32], or our proposed MADM feature extraction block."  We will modify it to "Ablation Study on the Feature Extraction Module: Substitution of Two-layer CNN in SRB with MADM Block, ConvNeXt Block, and Halo Block."

Thank you for bringing this to our attention, and we will ensure that the revised version accurately reflects the objective of the ablation experiments. We value the reviewer's feedback and are committed to enhancing the clarity and presentation of our work.


>2.In the training stage, is the the only loss function? I have not seen the formula of the total loss, but in the figure 2, you marked out three losses.

As depicted in Figure 2, we indeed utilize multiple loss functions in our approach. However, due to space limitations, we had to include the loss specific to our proposed method in the main text, while the remaining losses based on the baseline are presented in the supplementary material. We have addressed this particular situation in the concluding paragraphs of sections 3 and 4. In the revised version, we will highlight this statement to assist readers in locating the definitions of these losses in the appendix. For a more comprehensive understanding, please refer to Appendix B: "All Loss Functions in this Paper," as it contains detailed information regarding the various loss functions employed in our study.

>3.Is the formula in section 3.1 wrong? does not contain in 3.1, but the formula of MSRB in 3.2 contains BLSTM. So which one is right? 

Thank you for bringing up this question. The formula itself in section 3.1 is correct, but there was an issue with the wording in the last sentence of the first paragraph that led to your misunderstanding. Let me clarify the formulas:

The formula analysis is as follows:

In Equation (1):
$$I_{SR}= f_{pam}(f_{ps}(f_{blstm}([f_{b}(STN(I_{LR})),h_{t}]))))$$

In Equation (2):
MSRB(i_{lr},h_{t})=BLSTM([MLP(i_{lr}+MADM(i_{lr})),h_{t}])

Both formulas involve the extraction of features from the low-resolution input using the MSRB with residual structures. Then, the extracted features are combined with textual cues and fed into a bidirectional LSTM for integrating image features with text information. The function $f_{blstm}$ in Equation (1) corresponds to the $BLSTM$ in Equation (2). The function $f_{b}$ in Equation (1) is further elaborated as $\boldsymbol{MLP}(\boldsymbol{i}_{lr}+\boldsymbol{MADM}(\boldsymbol{i}_{lr}))$ in Equation (2).

The inconsistency in symbol usage between the formulas caused confusion in understanding. I will make the representations clearer by splitting the formula in section 3.1 into two parts: $f_{msrb}（I_{LR},h_{t}）=f_{blstm}([f_{b}(STN(I_{LR})),h_{t}]))$ and $I_{SR}= f_{pam}(f_{ps}(f_{msrb}（I_{LR},h_{t}）))$.

Furthermore, in the last sentence of the second paragraph in section 3.1, where it currently says, "The feature then enters the MSRB $f_b$", we will revise it to "Subsequently, the features are fed into the feature extraction module $f_b$ for further feature extraction." This change will help clarify the role of the module in the context.


>4.There are too many spelling mistakes in the paper.

We sincerely apologize for the presence of spelling mistakes in the paper and any instances where the descriptions may have been unclear or inaccurate. We recognize the criticality of maintaining linguistic excellence in scholarly works and are fully committed to rectifying these issues. In our forthcoming revision, we will conduct a thorough proofreading of the entire manuscript, diligently correcting any spelling errors and refining the language to enhance clarity and precision. We deeply value your feedback, as it assists us in improving the overall quality and clarity of our research.


# For Reviewer fsQx

## Comment：
We thank the reviewer for their detailed feedback. Please find detailed responses in the following.

>1 Novelty claim and motivation
>(1)Using Local Graph attention after pixel shuffling is interesting. However, the novelty of this design (also the mlp-based MSRB block) seems a little weak. The paper claims to propose these blocks (abstract and introduction). However, it seems more like introducing and utilizing existing blocks.
>(2) The motivation of MLP and graph-attention is unclear.
>In section 3.2, the paper says “MLP with lower inductive bias is able to learn from these auxiliary models more effectively.”
>I agree MLP and attention have lower inductive bias than convolution. However, in the text generation, it seems inductive bias is wanted since there are only 26 letters. For example, an inductive biased model can better remember these letters to achieve better results.

***These problems exhibit some interdependencies, which we will address by decomposing them into three subproblems for resolution.***

***Q1: Motivation and novelty of the article.***
![Motivation of our paper](https://anonymous.4open.science/r/RTSRN-FD14/motivation.png)


The main motivation of this article is to address the issue of image quality degradation caused by existing super-resolution models during the upsampling process. We illustrate this problem using a diagram (recreated as Figure 1 in our paper). The upsampling based on pixel shuffle, as described, combines neighboring features from low-resolution pixels using a CNN, expands the dimensions, and then rearranges the channels by grouping them. This process extends a pixel from the low-resolution image to a group of pixels in the high-resolution image. However, the repeated resampling and rearrangement of pixels lead to pixel distortion in the super-resolved image. Particularly, in high-frequency regions of the image (such as object edges), information loss or jagged edges may occur. Using the example provided in the paper, when two adjacent pixels aggregate neighboring features through a convolutional kernel with a size of 3 and stride of 1, the upsampled region is reconstructed from the information within a 4x3 region in the low-resolution image. This inevitably results in pixel distortion, especially at object edges. Existing works have overlooked this issue and mainly focused on alleviating the problem through feature extraction and upsampling algorithms. To the best of my knowledge, this paper introduces the novel concept of the "pixel-adapter," a post-processing module based on feature adaptive adjustment, to address the degradation in image quality caused by upsampling for the first time. This is the innovative aspect of the viewpoint presented in the paper. To mitigate pixel distortion, we propose an efficient graph attention mechanism that is agnostic to the adjacency matrix of the image. It integrates the high-order information of pixels that have undergone CNN interaction, uses graph attention to focus on both the pixel itself and its neighboring pixels, and adaptively adjusts the pixel features, thereby alleviating pixel distortion caused by upsampling. Additionally, considering the loss of high-frequency information caused by upsampling and the potential oversmoothing of graph attention, we utilize the gradient frequency histogram as a training objective to compel the model to focus on high-frequency details in the image.

***Q2: Motivation and novelty of the MLP design.***

In text image super-resolution tasks, feature extraction from text images serves as the foundation for subsequent processing. Compared to general images, text images contain textual information that can be effectively captured using both horizontal and vertical operators, as the text exists predominantly in those directions. Additionally, since text is a sequential information, there exists strong correlation between characters in terms of their spatial arrangement and context. To capture these interdependencies among features, we employ bidirectional LSTM.

Moreover, in our approach, we incorporate the Content-Aware Module (referred to as Text Perception Loss) proposed by STT[1]. This module utilizes a pre-trained Transformer to perceive the textual regions within the text image, acting as a training objective that compels the model to focus on the areas containing text. This process can be seen as a form of knowledge distillation, where the pre-trained Transformer serves as the teacher model and guides our model (referred to as the student model) to attend to the text regions in the image. By leveraging the reduced inductive bias of MLP compared to CNN and attention-based modules, our model is able to acquire more knowledge from the pre-trained model. Our experimental results, as shown in Table 7, provide empirical evidence supporting this claim. Following these principles, we design the MLP-based Sequential Residual Block (MSRB).


***Q3: Motivation and novelty of the graph attention mechanism.***

The motivation for using graph attention in this paper is described as follows. The pixel distortion issue caused by upsampling in image super-resolution mainly arises from the deformation and lack of smoothness in the detailed structure of target edge pixels. Global attention [2] struggles to focus on local regions' inconsistencies, and it incurs significant computational and memory overhead, resulting in slow processing speeds. Moreover, it also emphasizes relationships between blocks, which is not suitable for image super-resolution tasks due to inadequate granularity.

Block-wise attention [3] reduces memory consumption but exhibits an unbiased attention that only considers interactions among points within a region, lacking the desired granularity. On the other hand, pixel-wise graph attention models pixels as nodes on a graph and enables each pixel to interact with its neighboring pixels using the graph structure. Existing pixel-wise graph attention mechanisms [4] still employ adjacency matrices for interactions, where $X^{t+1}=\tilde{A}X^t$, with $X$ representing node features and $\tilde{A}$ denoting the attention-based adjacency matrix, which better represents relationships among neighbors. This approach effectively captures local structural information and leverages attention to update node features, offering a perfect solution to pixel distortion problems.

However, in images, neighbors are considered as the surrounding pixels within a one-pixel margin, represented by an adjacency matrix. The adjacency matrix obtained from images is particularly sparse, leading to significant memory consumption and slower processing speeds for this module. To address this issue, we propose an innovative sliding window-based graph attention mechanism specifically designed for images, which is independent of adjacency matrices. In this mechanism, neighbors in the image are defined as the surrounding pixels within a one-pixel margin, forming a 3x3 matrix with the central point as the target node and the surrounding points as neighbors (8-neighbor). Thus, a sliding window enables interactions between a node and its neighbors, and it facilitates parallel computation. As shown in Figure 7 of the original paper, this approach significantly reduces memory usage while greatly improving processing speed.

[1] Scene Text Telescope: Text-Focused Scene Image Super-Resolution
[2] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
[3] Scaling Local Self-Attention for Parameter Efficient Visual Backbones
[4] Pixel-wise Graph Attention Networks for Person Re-identification


>Result
>(1)What is the upscaling factor r in this paper? Since the graph-attention is conducted in upscaling. If experiments show that Pixel Adapter has more contribution in a larger upscaling factor than a smaller one, the effectiveness of Pixel Adapter can be more convincing.


In the field of text super-resolution, considering the small image size (32x128), we focused only on X2 super-resolution. This will be clearly indicated in the paper. To test the performance of different upscaling factors, while considering the model's computational efficiency, we conducted experiments using the TSRN model. The pixel adapter consists of the PAM and LCA modules, which were added to TSRN to evaluate their impact on performance. All experiments were validated using CRNN on TextZoom. The results are shown in the following table:


| model         | Upscaling Factor | Accuracy | PNSR | SSIM |
| :-----------: | :--------------: | :------: | :--: | :--: |
| TSRN          | 2                | 41.4%    | 20.67    | 0.7517    |
| TSRN+PAM+LCA  | 2                | **42.4%**       | **20.71**    | **0.7620**   |
| TSRN          | 4                | 27.8%       | 19.70    | 0.7117    |
| TSRN+PAM+LCA  | 4                | 30%        | 18.63    |  0.6714   |

After incorporating the PAM and LCA modules, the model consistently achieves a 1% higher recognition accuracy compared to the baseline. However, increasing the upsampling rate results in a significant decline in overall performance for both models. Notably, the model enhanced with our modules outperforms the model without them by 2.2%. This suggests that our model not only demonstrates superior performance but also exhibits enhanced robustness.

Moreover, the image quality metrics perform poorly after upsampling, potentially due to the interference between the gradient profile loss in TSRN [1] and our LCA module. We will include this experiment in the appendix for supplementary analysis.

[1] Scene Text Image Super-Resolution in the Wild


>(2)How is this method compared to the local attention-based super-resolution network, such as SwinIR. If training SwinIR from scratch is unaffordable, we can finetune a checkpoint pretrained on DIV2K in the same dataset and compare the result in the paper with the finetuned result. It is better to provide comparison with general super resolution

This paper primarily presents a post-processing module to alleviate the quality degradation caused by upsampling in image super-resolution. Consequently, we incorporated the post-processing modules, PAM and LCA loss, into ELAN [5] (a more efficient and performance-comparable model to SwinIR). Real-world super-resolution tasks were conducted using the same settings with an upsampling factor of r=2, and the results are presented in the following table:

| Method          | Set5          | Set14         | B100          | Urban100      | Manga109      |
| :-------------: | :-----------: | :-----------: | :-----------: | :-----------: | :------------ |
| TSRN            | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     |
| ELAN-light      | 38\.17/0.9611 | 33\.94/0.9207 | 32\.30/0.9012 | 32\.76/0.9340 | 39\.11/0.9782 |
| ELAN-light +LCA | 38\.18/0.9615 | 33\.97/0.9219 | 32\.34/0.9020 | 32\.80/0.9351 | 39\.14/0.9783 |
| ELAN-light +PAM | ***38\.23/0.9615*** | ***33\.99/0.9219*** | ***32\.35/0.9020*** | ***32\.88/0.9354*** | ***39\.21/0.9784*** |


Based on the experimental results, both our proposed LCA and PAM modules demonstrate significant improvements in general image super-resolution. Particularly, the PAM module exhibits a substantial enhancement in terms of PSNR. For instance, compared to ELAN-light, it achieved a PSNR improvement of 0.12 on the Urban100 dataset. It is worth noting that the model incorporating a combination of PCA and LCA is still under training on the server and has not yet converged.

[5] Efficient Long-Range Attention Network for Image Super-resolution

>(3) There is a new baseline in 2023. Improving Scene Text Image Super-Resolution via Dual Prior 
Modulation Network. Can this method compare with the DPMN result.

DPMN uses overall accuracy as the performance metric, while we utilize the Average Accuracy calculation method. After converting our performance metrics to the overall accuracy format, the results are presented in the following table:

| Method    | CRNN  |        |       |         | MORAN  |        |        |         | ASTER  |        |        |         |
| :-------: | :---: | :----: | :---: | :-----: | :----: | :----: | :----: | :-----: | :----: | :----: | :----: | :-----: |
|           | Easy  | Medium | Hard  | Average | Easy   | Medium | Hard   | Average | Easy   | Medium | Hard   | Average |
| TATT+DPMN | 64\.4 | 54\.2  | **39\.2** | 53\.4   | 73\.26 | **61\.45** | 43\.86 | 60\.42  | 79\.25 | **64\.07** | 45\.20 | 63\.89  |
| RTSRN     | **65\.6** | **55\.4**  | 38\.8 | **54\.1**   | **75\.4**  | 61\.0  | **44\.2**  | **61\.2**   | **79\.8**  | 61\.7  | **47\.1** | **63\.9**  |


In terms of accuracy performance metrics, our method demonstrates higher accuracy. However, when examining subcategories in detail, the use of a more general pixel-based refinement approach, which lacks sufficient exploration of semantic information, may lead to relatively lower accuracy for super-resolved difficult samples. Nevertheless, our method is more versatile and applicable to general image super-resolution tasks.

# For Reviewer ntsv

## Comment：
We thank the reviewer for their detailed feedback. Please find detailed responses in the following.

>1.It is well known that self-attention is a common module, but the article did not explain why it is effective for upsampling? Meanwhile, the time and memory usage of the three proposed modules global-wise, block-wise, and pixel-wise have not been experimentally tested in this article.


 The pixel distortion issue caused by upsampling in image super-resolution mainly arises from the deformation and lack of smoothness in the detailed structure of target edge pixels. Global attention [1] struggles to focus on local regions' inconsistencies, and it incurs significant computational and memory overhead, resulting in slow processing speeds. Moreover, it also emphasizes relationships between blocks, which is not suitable for image super-resolution tasks due to inadequate granularity.

Block-wise attention [2] reduces memory consumption but exhibits an unbiased attention that only considers interactions among points within a region, lacking the desired granularity. On the other hand, pixel-level graph attention models pixels as nodes on a graph and enables each pixel to interact with its neighboring pixels using the graph structure. Existing pixel-level graph attention mechanisms [3] still employ adjacency matrices for interactions, where $X^{t+1}=\tilde{A}X^t$, with $X$ representing node features and $\tilde{A}$ denoting the attention-based adjacency matrix, which better represents relationships among neighbors. This approach effectively captures local structural information and leverages attention to update node features, offering a perfect solution to pixel distortion problems.

However, in images, neighbors are considered as the surrounding pixels within a one-pixel margin, represented by an adjacency matrix. The adjacency matrix obtained from images is particularly sparse, leading to significant memory consumption and slower processing speeds for this module. To address this issue, we propose an innovative sliding window-based graph attention mechanism specifically designed for images, which is independent of adjacency matrices. In this mechanism, neighbors in the image are defined as the surrounding pixels within a one-pixel margin, forming a 3x3 matrix with the central point as the target node and the surrounding points as neighbors (8-neighbor). Thus, a sliding window enables interactions between a node and its neighbors, and it facilitates parallel computation. As shown in Figure 7 of the original paper, this approach significantly reduces memory usage while greatly improving processing speed.

[1] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

[2] Scaling Local Self-Attention for Parameter Efficient Visual Backbones

[3] Pixel-wise Graph Attention Networks for Person Re-identification


 **For the problem of the time and memory usage**, the Figure 7 in the paper compares the time and memory usage of different attention mechanisms. It presents a line graph that illustrates two aspects: on one hand, it compares the pixel-level graph attention (PGA) with the block-wise attention, represented as "Halo" in the graph. However, the corresponding descriptions in the figure are unclear and require clarification. We will make modifications to address this issue and also include tests for global attention. The details are as follows:

 waitting


 >2.The PSNR and SSIM values of PAM in both Table 7 and Table 4 are the same, but the ACC values are different. Can you explain why?

The transcription error regarding the accuracy in Table 7 has been corrected. The accuracy value of 52.3 is indeed accurate. I have double-checked all the data and confirmed that there are no other errors. The training log files for the table 1 results are attached in supplementary materials providing a traceable source for the results.

>3.PAM needs to conduct separate ablation experiments on the super-resolution module, and compare it with various upsampling methods such as shuffle pixel. In comparative experiments, it is necessary to provide metrics for a number of parameters and FLOPS.

In order to validate the effectiveness of the PAM as a post-processing module for refining super-resolution images after Pixel Shuffle, we tested different post-processing modules, including a simple feature processing module CNN, the block-wise attention module Halo, and our proposed PAM, on top of the baseline model. The results, As shown in the table below,, demonstrate significant improvements in both recognition accuracy and image quality metrics with the PAM. We also conducted tests to measure the parameters and FLOPS. We compared the results of the model on the text recognition task using the CRNN. The results are presented below:

| Method        | Parameter | FLOPs | Accuracy | PNSR   | SSIM    |
| :-----------: | :-------: | :---: | :------: | :----: | :-----: |
| Baseline      |      17.15 M      |   2.52 GMac    | 51\.7%    | 20\.56 | 0\.7481 |
| Baseline+Halo |    17.16 M        |   2.57  GMac   | 51\.5%    | 19\.62 | 0\.7426 |
| Baseline+CNN  |      17.19 M     |     2.67  GMac  | 51,6%     | 20\.86 | 0\.7567 |
| Baseline+PAM  |      17.16 M      |   2.57 GMac    | 52\.3%    | 20\.97 | 0\.7628 |

By incorporating the PAM into our model, we observed that the increase in parameters and FLOPs is comparable to, or even less than, the increase caused by the halo effect. This is primarily due to the pixel-level attention mechanism based on a sliding window and an adjacency matrix that we have proposed, which is independent of the pixels. The increase in parameters is primarily attributed to the linear layers corresponding to the queries (q), keys (k), and values (v) in the attention mechanism. As the PAM only considers local neighbors, its computational complexity is essentially similar to that of the halo effect. 

We compared the impact of different upsampling modules on the experiments, including Pixel Shuffle, Bilinear, and Transposed Convolution. The experimental results are shown below:

| Method                        | Parameter | FLOPs | Accuracy | PNSR | SSIM |
| :---------------------------: | :-------: | :---: | :------: | :--: | :--: |
| RTSRN -Bilinear Interpolation |     17.03 M      |    2.26 GMac   |  49.4%        | 19.25     | 0.7341     |
| RTSRN -Deconvolution          |     17.03 M      |   2.64 GMac    |    48.8%      |  20.64  |   0.74325 |
| RTSRN -pixel shuffle          |     17.03 M       |   2.41 GMac    |53.3%       |    20.16     |   0.7619      |

According to the experimental results, it has been observed that the pixel shuffle-based upsampling module achieves higher accuracy and better image quality.

>4.Although the PAM module is new, self-attention is more common. MLP-based sequential residual block is not very innovative. The MLP module proposed in [1] is similar to the MLP module in this article? HOG references the [2] and lacks more ablation experiments on PAM, such as verifying its effectiveness by adding it to other super-resolution modules.


***The main differences can be explained as follows:***

In text image super-resolution tasks, feature extraction from text images serves as the foundation for subsequent processing. Compared to general images, text images contain textual information that can be effectively captured using both horizontal and vertical operators, as the text exists predominantly in those directions. Additionally, since text is a sequential information, there exists strong correlation between characters in terms of their spatial arrangement and context. To capture these interdependencies among features, we employ bidirectional LSTM.

Moreover, in our approach, we incorporate the Content-Aware Module (referred to as Text Perception Loss) proposed by STT[1]. This module utilizes a pre-trained Transformer to perceive the textual regions within the text image, acting as a training objective that compels the model to focus on the areas containing text. This process can be seen as a form of knowledge distillation, where the pre-trained Transformer serves as the teacher model and guides our model (referred to as the student model) to attend to the text regions in the image. By leveraging the reduced inductive bias of MLP compared to CNN and attention-based modules, our model is able to acquire more knowledge from the pre-trained model. Our experimental results, as shown in Table 7, provide empirical evidence supporting this claim. Following these principles, we design the MLP-based Sequential Residual Block (MSRB).

We are conducting experiments to verify the effectiveness of our module by integrating it into ELAN-light. The results of the experiments have demonstrated that our module effectively enhances the image quality of super-resolution images. For more detailed information, please refer to the answer to the following question.

[1] Scene Text Telescope: Text-Focused Scene Image Super-Resolution

>5.Although PAM is important in this article, there are no further experiments to prove its role in super-resolution. In previous work [3], a series of comparative experiments were basically conducted on super-resolution models.


We incorporated the post-processing modules, PAM and LCA loss, into ELAN [1] (a more efficient and performance-comparable model to SwinIR). Real-world super-resolution tasks were conducted using the same settings with an upsampling factor of r=2, and the results are presented in the following table:

| Method          | Set5          | Set14         | B100          | Urban100      | Manga109      |
| :-------------: | :-----------: | :-----------: | :-----------: | :-----------: | :------------ |
| TSRN            | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     | PSNR/SSIM     |
| ELAN-light      | 38\.17/0.9611 | 33\.94/0.9207 | 32\.30/0.9012 | 32\.76/0.9340 | 39\.11/0.9782 |
| ELAN-light +LCA | 38\.18/0.9615 | 33\.97/0.9219 | 32\.34/0.9020 | 32\.80/0.9351 | 39\.14/0.9783 |
| ELAN-light +PAM | ***38\.23/0.9615*** | ***33\.99/0.9219*** | ***32\.35/0.9020*** | ***32\.88/0.9354*** | ***39\.21/0.9784*** |

Based on the experimental results, both our proposed LCA and PAM modules demonstrate significant improvements in general image super-resolution. Particularly, the PAM module exhibits a substantial enhancement in terms of PSNR. For instance, compared to ELAN-light, it achieved a PSNR improvement of 0.12 on the Urban100 dataset. It is worth noting that the model incorporating a combination of PCA and LCA is still under training on the server and has not yet converged.

[1] Efficient Long-Range Attention Network for Image Super-resolution




# For Reviewer ntsv

## Comment：
We thank the reviewer for their detailed feedback. Please find detailed responses in the following.


>1.The authors claim that they use graph neural networks, however, as Algorithm 1 demonstrates, the proposed PAM performs the attention mechanism within a local window and does not perform graph neural networks related operation. A graph is an irregular topology and performing attention mechanisms on regular pixels cannot be called graph attention.

In PGANet [1], the pixels in the image are treated as nodes in a graph, where the surrounding pixels are considered as neighbors. They utilize adjacency matrices for interactions, represented as $X^{t+1}=\tilde{A}X^t$, where $X$ represents node features and $\tilde{A}$ denotes the attention-based adjacency matrix. This approach effectively captures local structural information and utilizes attention to update node features, providing a solution to pixel distortion problems. However, the adjacency matrix obtained from the image is particularly sparse, resulting in significant memory consumption and slower computations for this module. To address this issue, we propose an innovative graph attention mechanism based on sliding windows that is independent of adjacency matrices and suitable for regular image graphs. In this mechanism, the neighbors of each pixel are considered as the surrounding pixels within a 3x3 matrix, where the center pixel serves as the target node, and the surrounding pixels act as neighbors (8-neighbors). Instead of representing this relationship using adjacency matrices, we utilize sliding windows. Each window facilitates interactions between a node and its neighbors, enabling parallel computation. Specifically, in Algorithm 1, the variable "q" represents the set of nodes in the graph with dimensions [b, n, 1, c], while "k" represents the corresponding neighbors for each node (including the node itself). Subsequently, we calculate attention using "q" and "k" to obtain an attention matrix "att" with dimensions [b, n, 1, k*k]. This attention matrix can be viewed as a specialized adjacency matrix $\tilde{A}$. With the help of the attention matrix, we efficiently propagate the features of pixels in this graph, as shown in Figure 7 of the original paper. Our method significantly improves computational speed while reducing memory usage.

[1] Pixel-wise Graph Attention Networks for Person Re-identification


>2.The authors have added new modules to the previous sota model C3-STISR, and the authors need to demonstrate that the performance improvement is not due to the addition of more model parameters.




| Method        | Parameter | FLOPs | Accuracy | PNSR   | SSIM    |
| :-----------: | :-------: | :---: | :------: | :----: | :-----: |
| Baseline      | 17.15 M      |   2.52 GMac| 51\.7%    | 20\.56 | 0\.7481 |
| Baseline+MSRB |   17.03 M         |     2.41 GMac  |   52.8%       |    20.85     |    76.05     |
| Baseline+PAM  |  17.16 M      |   2.57 GMac    | 52\.3%    | 20\.97 | 0\.7628 |
| Baseline+LCA  |     17.15 M       |     2.52 GMac  |     52.5%     |    20.72    |     76.07     |


According to the findings presented in above Table, the effectiveness of our modules is evident, as they contribute to improved performance with only a slight increase in model complexity. This is primarily due to the pixel-level attention mechanism based on a sliding window and an adjacency matrix that we have proposed, which is independent of the pixels. The increase in parameters is primarily attributed to the linear layers corresponding to the queries (q), keys (k), and values (v) in the attention mechanism. As the PAM only considers local neighbors, its computational complexity is essentially similar to that of the halo effect. MSRB is composed of axis-aligned MLP, leading to a substantial reduction in both the parameter count and computational complexity of the model.

In the following table, we present a comparison of the effects of different post-processing modules on performance, parameter count, and FLOPs. The results clearly indicate that the increase in model parameters is primarily driven by the necessity to process larger images as a result of the post-processing modules. Remarkably, the employment of a CNN as the post-processing module leads to a greater increase in parameter count and FLOPs compared to utilizing PAM.

| Method        | Parameter | FLOPs | Accuracy | PNSR   | SSIM    |
| :-----------: | :-------: | :---: | :------: | :----: | :-----: |
| Baseline      |      17.15 M      |   2.52 GMac    | 51\.7%    | 20\.56 | 0\.7481 |
| Baseline+Halo |    17.16 M        |   2.57  GMac   | 51\.5%    | 19\.62 | 0\.7426 |
| Baseline+CNN  |      17.19 M     |     2.67  GMac  | 51,6%     | 20\.86 | 0\.7567 |
| Baseline+PAM  |      17.16 M      |   2.57 GMac    | 52\.3%    | 20\.97 | 0\.7628 |



>3.In the paper the authors only demonstrate that the proposed local attention aggregated pixel features are superior in efficiency compared to PGA and Halo. The authors should give the increased cost (e.g. FLOPs and #params) of the proposed method compared to baseline methods (e.g. C3-STISR).


As show in below table, we compare increase cost of the proposed method compared to baseline.

| Method        | Parameter | FLOPs | Accuracy | PNSR   | SSIM    |
| :-----------: | :-------: | :---: | :------: | :----: | :-----: |
| Baseline      | 17.15 M      |   2.52 GMac     | 51\.7%    | 20\.56 | 0\.7481 |
| RTSRN         |    17.03 M        |    2.41 GMac   |   53.3%       |    20.16     |   0.7619      |




>4.In Table 1, the authors should give a reasonable explanation as to why using the proposed method with more parameters leads to a sub-optimal accuracy compared with TATT when Aster is used as the recognizer.

This discrepancy is due to differences in experimental settings. We refer to the experimental setup of C3-STISR, where during the model training phase, CRNN is used to evaluate the text recognition accuracy on the super-resolved text images. This allows the model to better adapt to the CRNN recognizer. On the other hand, in TATT, Aster is used as the recognizer during the testing phase. The recognition results of MORAN and ASTER in Table 1 are obtained by directly recognizing the super-resolved images generated by training CRNN models. To ensure fairness, we examine the recognition results of MORAN on the easy, medium, and hard datasets in Table 1, and compare them with TATT. Our model achieves improvements of 2.9%, 0.8%, and 1.1% in accuracy, respectively. These results demonstrate the superiority of our approach.



>5.Finally, don't forget to fix the following typo: a.The latex template should show line numbers during the blind review. b. The case of each section heading needs to be consistent (e.g., “2.1 Vision backbone” is not consistent with “2.2 Scene Text Image Super-Resolution”). c. In the second point of the contuibution summary, “We propose an MSRB ..” d. The presentation quality of Figure 2 should be further improved. e. The meaning of C is inconsistent in the second and third paragraphs of Sec. 3.1. f. First paragraph in the Sec. 3.3, “followed by…”.


We have standardized the titles to have all words capitalized. Figure 2 will be replaced with a more detailed image. In Section 3.1, the first 'C' will be changed to '3', and the modified formula will be: $I_{LR}\in\mathbb{R}^{H\times W\times 3}$. The first letter in the first paragraph of Sec. 3.3 has already been capitalized. We greatly appreciate the thorough review of our paper by the reviewers and their valuable feedback regarding spelling errors and formatting issues. The necessary modifications have already been made based on the issues you pointed out. 

