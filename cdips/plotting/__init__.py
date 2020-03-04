from datetime import datetime

def savefig(fig, figpath):

    if not figpath.endswith('.png'):
        raise ValueError('figpath must end with .png')

    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

    pdffigpath = figpath.replace('.png','.pdf')
    fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=450)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), pdffigpath))
