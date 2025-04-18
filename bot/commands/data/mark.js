const { SlashCommandBuilder } = require('discord.js');
const axios = require('axios');
const fs = require('fs');


// Command data.
const data = new SlashCommandBuilder()
    .setName('mark')
    .setDescription('Classify a message to a specific category')
    .addStringOption(option => 
        option.setName("category")
        .setDescription("The category to classify the message")
        .setRequired(true)
    )
    .addStringOption(option =>
        option.setName("message_id")
        .setDescription("The message ID to classify")
        .setRequired(false)
    )


// Command execution function.
async function execute(message) {
    const category = message.options.getString('category');
    const message_id = message.options.getString('message_id') ?? null;

    // get target message data
    let target_message = null

    // if no message id is provided, get the last non-bot message in the channel
    if (!message_id) {
        //get last non-bot message
        target_message = await message.channel.messages.fetch({ limit: 10, before: message.id }).then(messages => {
            let non_bot_messages = messages.filter(m => !m.author.bot);
            if (non_bot_messages.size > 0) {
                return non_bot_messages.first();
            } else {
                return message.reply("I couldn't find any non-bot messages in this channel. Please provide a valid message ID.");
            }
        })
        if (!target_message) return;

    } else { //otherwise, get the message straight by ID.
        target_message = await message.channel.messages.fetch(message_id).catch(err => {
            message.reply("I couldn't find that message. Please provide a valid message ID.");
        })
        if(!target_message) return;
    }

    if (category == null || target_message == null) {
        return message.reply("Please provide a valid category and message ID.");
    }

    // just to make sure we don't make a discord channel that has weird characters in it
    //the regex allows only alphabetic characters and underscores
    if (!/^[a-zA-Z0-9_]+$/.test(category)) {
        message.reply("Please provide a valid category. Only alphanumeric characters and underscores are allowed.");
        return;
    }

    // send data to API
    let url = process.env.API_URL + '/message'
    let data = {
        message: target_message.content, //the content of the message
        category: category, //the target category
        user: target_message.author.id, //the id of the user who sent the message
        guild: target_message.guild.id, //the id of the guild
        timestamp: target_message.createdTimestamp.toString() //the timestamp of the message (idk if we need this, but just in case)
    };

    // axios post - make a post request to the API with the data
    await axios.post(url, data).then(res => {
        if (res.status == 200) {
            message.reply('Message classified successfully!');
        } else {
            return message.reply('Error classifying message: ' + res.statusText);
        }
    }).catch(err => {
        console.log(err);
        message.reply('Error classifying message: ' + err.message);
    });


    // update the categories discord category
    let discord_category = process.env.NOTES_CATEGORY_ID;
 
    // get the discord category channel (the channel where the categories are)
    let discord_category_channel = await message.guild.channels.fetch(discord_category).catch(err => {
        return message.reply("I couldn't find that category. Please provide a valid category ID.");
    });

   

    //get the current existing categories in the spreadsheet from the API and delete any channels that are not in the categories
    let category_url = process.env.API_URL + '/categories/' + message.guild.id;
    await axios.get(category_url).then(res => {
        //delete any channels that are not in the categories
        let categories = res.data.categories;
        let channels = discord_category_channel.children;

        channels.cache.forEach(channel => {
            if (!categories.includes(channel.name)) {
                channel.delete().catch(err => {
                    console.log(err);
                    message.reply('Error deleting channel: ' + err.message);
                });
            }
        });

        //create channels for each category if they do not exist
        channels = discord_category_channel.children;
        channels = channels.cache.map(channel => channel.name);
        categories.forEach(category => {
            if (!channels.includes(category)) {
                discord_category_channel.guild.channels.create({
                    name: category,
                    type: 0,
                    parent: discord_category_channel.id
                }).catch(err => {
                    console.log(err);
                    message.reply('Error creating channel: ' + err.message);
                });
            }
        });
    }).catch(err => {
        console.log(err);
        message.reply('Error getting categories: ' + err.message);
    })


    //wait 5 seconds to make sure the channel is created
    await new Promise(resolve => setTimeout(resolve, 5000));


    //send copied message to category channel
    discord_category_channel = await message.guild.channels.fetch(discord_category)
    let category_channel = discord_category_channel.children.cache.find(channel => channel.name === category);
    if (category_channel) {
        category_channel.send(`Marked by <@!${message.member.id}>\nMessage source: ${target_message.url}\n\n${target_message.content}`);
    } else {
        message.channel.send('Error: Category channel not found.');
    }

}

module.exports = {
    data: data,
    execute: execute
}

