FROM php:8.1-apache

WORKDIR /var/www/html

COPY ./.htaccess ./
COPY ./*.md ./
COPY ./*.php ./
COPY ./util ./util
COPY ./static ./static

# Install PHP extensions
RUN docker-php-ext-install pdo pdo_mysql mysqli

# Enable Apache mod_rewrite (common for frameworks like Laravel, WordPress)
#RUN a2enmod rewrite

# Clone your GitHub repository into Apache root
# ⚠️ You may want to copy from local clone instead if auth is required
#RUN apt-get update && apt-get install -y git \
#    && git clone https://github.com/your-username/your-repo.git /var/www/html \
#    && chown -R www-data:www-data /var/www/html
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y git

RUN chown -R www-data:www-data /var/www/html

# Optional: Configure Apache settings if needed
# COPY ./custom-apache.conf /etc/apache2/sites-available/000-default.conf
