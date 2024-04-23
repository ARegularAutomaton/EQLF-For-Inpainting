# Equivariant learning framework for inpainting tasks
## Model Training
To train models, run `Python train.py --problem <problem> --localised <localised> --closure <closure> --epochs <epochs> --transform <transform>`. 

`<problem>` takes values `inpaint` or `ct`: 
- If set to `inpaint`, the model is trained under the inpainting problem setting
- Otherwise that of Sparse-view CT. Note that only algorithms (closures) 1, 2 and 6 has CT implementations (using astra-toolbox), but can only back-propagate the equivariance loss $\mathcal{L}_{EQ}$, due to incompatibility of astra-toolbox with PyTorch. Use the training script in repository [REI-CT]() for training models for CT tasks.

`<localised>` takes values `true` or `false`: 
This parameter is applicable only when the `<problem>` is set to `inpaint`. Defaulted to `false`.
- If set to `true`, the model is trained under the localised inpainting problem setting
- Otherwise that of distributed inpainting. 

`<closure>` takes values `1`-`7`: 
- `1` trains the model using the supervised loss with equivariance enforced
- `2` trains the model with the EI framework
enforced
- `3` trains the model without supervision adverserially to condition that the distribution of transformed images is close to that of reconstructed images as well as with measurement consistency and equivariance enforced
- `4` trains the model with measurement consistency loss alone
- `5` trains the model with the REI framework
- `6` trains the model with supervised loss on measurements with additive Gaussian noise
- `7` trains the model with supervised loss

`<epochs>` takes an integer:
- Setting this to some integer trains the model for that many epochs.

`<transform>` takes values 1-3:
- `1` trains the model using a group of permutations
- `2` trains the model using a group of shift transformations
- `3` trains the model using a group of rotations

## Model Evaluation
To evaluate trained models, run `Python test.py --problem <problem> --localised <localised> --epochs <epochs>`. The test script will look for models stored in directory `trained_model` with subdirectories matching the arguments given. By default, the trained models are saved to such a directory by the training script. 
