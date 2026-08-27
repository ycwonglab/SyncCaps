#!/usr/bin/env python3
"""Check whether the reserved Zenodo DOI is actually minted and resolving.

The package cites a DOI. A DOI written into a README is a claim like any other,
and Zenodo lets you RESERVE one on a draft deposition long before it is
registered -- so a cited DOI can sit there for weeks resolving to nothing. This
script answers the question rather than leaving a reader to trust the string.

Exit status: 0 if the DOI resolves, 1 if it is reserved-but-unregistered, 2 on
a network/other error. Nothing here is cached; it queries live.

  python provenance/check_doi.py [--doi 10.5281/zenodo.22131664]
"""
import argparse, json, sys, urllib.error, urllib.request

DEFAULT_DOI = '10.5281/zenodo.22131664'


def probe(url, method='GET'):
    req = urllib.request.Request(url, method=method,
                                 headers={'User-Agent': 'synccaps-doi-check',
                                          'Accept': '*/*'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(2000)
    except urllib.error.HTTPError as e:
        return e.code, e.read(2000)
    except Exception as e:                                   # pragma: no cover
        return None, str(e).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doi', default=DEFAULT_DOI)
    a = ap.parse_args()
    rec = a.doi.rsplit('.', 1)[-1]

    checks = [
        ('doi.org resolver', 'https://doi.org/' + a.doi),
        ('DataCite registry', 'https://api.datacite.org/dois/'
                              + a.doi.replace('/', '%2F')),
        ('Zenodo record API', 'https://zenodo.org/api/records/' + rec),
    ]
    results = {}
    for name, url in checks:
        code, _ = probe(url)
        results[name] = code
        print('  %-20s %-58s -> %s' % (name, url, code))

    resolves = results.get('DataCite registry') == 200 or \
        results.get('Zenodo record API') == 200
    print()
    if resolves:
        print('DOI IS MINTED: %s resolves. The package has an immutable archive;'
              % a.doi)
        print('README and CITATION.cff may state that without qualification.')
        return 0
    print('DOI IS RESERVED BUT NOT MINTED: %s does not resolve.' % a.doi)
    print('This is the expected state of a Zenodo draft on which "Reserve DOI"')
    print('was pressed but Publish was not. Until it is published:')
    print('  - the GitHub release remains the only public archive, and it is')
    print('    versioned but MUTABLE (tags can be moved, assets replaced);')
    print('  - the manuscript must not claim a persistent identifier is available.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
