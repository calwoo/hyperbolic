# Utility functions for Poincare embeddings

import nltk
import ssl

def nltk_download_ssl_override(corpus):
    """
    Overrides SSL certificate verification necessary for urllib. This is necessary when working
    with Mac OS X Catalina, since it alters OpenSSL permissions for many Python libraries.
    """

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download(corpus)



if __name__ == "__main__":
    nltk_download_ssl_override("wordnet")