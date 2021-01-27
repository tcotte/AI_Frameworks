import numpy as np
import torch


# Training
def train(model, train_dataloader, device, optimizer, epoch):
    print("<=========== EPOCH " + str(epoch + 1) + " ===========>")
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        if torch.cuda.is_available():
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        else:
            loss = model(torch.tensor(b_input_ids).to(device).long(), token_type_ids=None, attention_mask=b_input_mask,
                         labels=b_labels)
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # calculate average losses
        train_loss = tr_loss / nb_tr_steps

        # Display
        if nb_tr_steps % 30 == 1:
            print('\r Epoch: {} \tTraining Loss: {:.6f}'.format(str(epoch+1), train_loss))

    return train_loss


def predict(model, validation_dataloader, device):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # store logits and labels
    v_logits = []
    v_labels = []

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(torch.tensor(b_input_ids).to(device).long(), token_type_ids=None,
                           attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # store logits and labels
        v_logits.extend(logits)
        v_labels.extend(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

        # calculate average test accuracy
    eval_acc = eval_accuracy / nb_eval_steps

    # Print training/validation statistics
    print('\tEvaluation accuracy: {:.3f}%'.format(eval_acc))

    return [v_logits, v_labels, eval_acc]


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
