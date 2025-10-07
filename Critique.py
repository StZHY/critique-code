import torch
from torch import nn
import utility.trainer as trainer
import utility.tools as tools
import utility.losses as losses



class Critique(nn.Module):
    def __init__(self, args, original_user_embedding_weights, original_item_embedding_weights):

        super(Critique, self).__init__()
        
        self.device = torch.device("cuda:"+str(args.gpu)) if args.cuda else torch.device("cpu")
        self.model_name = "Critique"
        self.activation = nn.Sigmoid()
        self.args = args

        self.user_embedding = nn.Embedding.from_pretrained(
            original_user_embedding_weights.clone().detach(), freeze=False
        )
        self.item_embedding = nn.Embedding.from_pretrained(
            original_item_embedding_weights.clone().detach(), freeze=True
        )
        
    def supplement_average_items(self, train_user_set):

        user_num = len(train_user_set)
        aver_items_tensor_all = torch.empty(user_num, int(self.args.embedding_size)).to(self.device)
        
        item_weights = self.item_embedding.weight

        for user, items in train_user_set.items():
            
            items_tensor = torch.tensor(items, dtype=int, device=self.device)
        
            # Look up the embeddings using the item_weights tensor directly.
            # This resolves the TypeError.
            item_emb = item_weights[items_tensor]
        
            # Calculate the mean of the embeddings along the first dimension.
            aver_items_tensor = torch.mean(item_emb, dim=0)
        
            # Assign the averaged tensor to the appropriate user index.
            aver_items_tensor_all[user] = aver_items_tensor

        combined_item_embedding = torch.cat((item_weights, aver_items_tensor_all), dim=0)
        self.aver_item_embedding = nn.Embedding.from_pretrained(combined_item_embedding, freeze=True).to(self.device)
        
        
    def forward(self, user, positive, bpr_negative, cl_negative):

        user_embed = self.user_embedding(user.long())
        positive_embed = self.aver_item_embedding(positive.long())
        bpr_negative_embed = self.item_embedding(bpr_negative.long())
        cl_negative_embed = self.item_embedding(cl_negative.long())

        bpr_loss = losses.get_critique_loss_base(user_embed, bpr_negative_embed)
        cl_loss = losses.get_critique_InfoNCE_loss(user_embed, positive_embed, cl_negative_embed)
        
        total_loss = bpr_loss + self.args.critique_cl_lambda * cl_loss

        return total_loss

    def get_rating_for_test(self, user):

        all_user_gcn_embed, all_item_gcn_embed = self.user_embedding.weight, self.item_embedding.weight

        user_gcn_embed = all_user_gcn_embed[user.long()]

        rating = self.activation(torch.matmul(user_gcn_embed, all_item_gcn_embed.t()))

        return rating